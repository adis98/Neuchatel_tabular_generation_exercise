import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import Preprocessor
from utils.model import DiffusionMLP
from utils.dic_experimenter import match_DICs_fast, precompute_cat_DICs_fast


def get_lr_lambda(current_epoch, num_train_epochs, num_warmup_epochs=150):
    if current_epoch < num_warmup_epochs:
        return current_epoch / max(1, num_warmup_epochs)
    else:
        return max(0.0, (num_train_epochs - current_epoch) / max(1, num_train_epochs - num_warmup_epochs))


def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prepper = Preprocessor(args.dataname)
    train_df = prepper.df_train
    train_data_encoded = prepper.encodeDfToNp(train_df)
    train_data_standardized = prepper.standardize_np(train_data_encoded)
    num_dim = len(prepper.num_cols)
    cat_dims = [len(cats) for cats in prepper.OneHotEncoder.categories_]
    x_train = torch.tensor(train_data_standardized, dtype=torch.float32)
    model = DiffusionMLP(in_dim=train_data_standardized.shape[1]).to(device)
    train_loader = DataLoader(
        x_train,
        batch_size=args.batch_sz,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: get_lr_lambda(ep, args.epochs))
    criterion1 = torch.nn.MSELoss(reduction='none')
    criterion2 = torch.nn.CrossEntropyLoss(reduction='none')
    model.train()
    pbar = tqdm(range(args.epochs), desc="Training")
    cat_feature_sizes = [prepper.OneHotEncoder.categories_[i].shape[0] for i in
                         range(len(prepper.OneHotEncoder.categories_))]
    replica_counts = torch.tensor(np.array(([1] * len(prepper.num_cols)) + cat_feature_sizes)).to(device)
    bucket_counts = (
            np.sum(train_data_standardized[:, len(prepper.num_cols):], axis=0) / train_data_standardized.shape[0])
    start = 0
    entropies = [1.0] * num_dim
    cat_bias_init = []
    for categ in prepper.OneHotEncoder.categories_:
        end = start + categ.shape[0]
        view = bucket_counts[start:end]
        entropies.append(np.sum(-view * np.log(view)))
        cat_bias_init.append(torch.tensor(np.log(view), dtype=torch.float32))
        start = end

    possible_timesteps = torch.linspace(0, 1, args.timesteps)
    sigmas_base = np.linspace(0, args.sigma_max, args.timesteps)
    cardinalities = cat_dims + [300]
    reference_DIC = precompute_cat_DICs_fast(sigmas_base, [2])
    if args.alignDIC:
        corrected_sigmas = match_DICs_fast(reference_DIC, sigmas_base, cardinalities)
        corrected_sigmas_num = corrected_sigmas[:, -1]
        corrected_sigmas = corrected_sigmas[:, :-1]
        corrected_sigmas = np.concatenate(
            (corrected_sigmas_num.reshape(-1, 1).repeat(num_dim, axis=1), corrected_sigmas), axis=1)

    else:
        corrected_sigmas = sigmas_base.reshape(-1, 1).repeat(num_dim + len(cat_dims), axis=1)
    for epoch in pbar:
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            indices = torch.randint(0, len(possible_timesteps), (batch.shape[0],))
            timesteps = possible_timesteps[indices]
            epsilons = torch.normal(0, 1, size=batch.shape).to(device)
            timesteps_t = timesteps.to(device)
            """Forward noising"""
            sigmas = torch.tensor(corrected_sigmas[indices], dtype=torch.float32).to(device)
            sigmas_expanded = sigmas.repeat_interleave(replica_counts, dim=1)

            """THE LINE BELOW IS INCORRECT; CORRECT IT!"""
            c_skip = 1/sigmas_expanded

            c_out = sigmas_expanded / ((1 + sigmas_expanded ** 2).sqrt())  # sqrt(1-alpha bar t)
            batch_noised = (batch + epsilons * sigmas_expanded).to(device)
            outs = model(torch.sqrt(c_skip) * batch_noised, timesteps_t)
            cont_preds = c_skip[:, :num_dim] * batch_noised[:, :num_dim] + c_out[:, :num_dim] * outs[:, :num_dim]
            x0_pred = torch.concat([cont_preds, outs[:, num_dim:]], dim=1)

            """THE LINE BELOW IS INCORRECT; CORRECT IT!"""
            loss = criterion1(x0_pred[:, :num_dim], batch_noised[:, :num_dim])

            cat_chunks_pred = torch.split(outs[:, num_dim:], cat_feature_sizes, dim=-1)
            cat_chunks_gt = torch.split(batch[:, num_dim:], cat_feature_sizes, dim=-1)
            for logit_chunk, target_chunk in zip(cat_chunks_pred, cat_chunks_gt):
                target_idx = target_chunk.argmax(dim=-1)

                """THE LINE BELOW IS INCORRECT; CORRECT IT!"""
                cat_loss = (criterion1(logit_chunk, target_idx)).unsqueeze(-1)

                loss = torch.concat((loss, cat_loss), dim=-1)

            denoiser_loss = loss.mean()
            denoiser_loss.backward()
            optimizer.step()
            total_loss += denoiser_loss.detach().cpu().numpy()
        scheduler.step()
        pbar.set_postfix(loss=total_loss)
    if args.save:
        models_dir = f'saved_models/{args.dataname}/'
        os.makedirs(models_dir, exist_ok=True)
        if args.alignDIC:
            torch.save(model.state_dict(), f'{models_dir}/model_aligned.pt')
        else:
            torch.save(model.state_dict(), f'{models_dir}/model_unaligned.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--batch_sz', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--sigma_max', type=float, default=100, help='last variance schedule')
    parser.add_argument('--save', type=bool, default=False, help='Save the trained model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--alignDIC', type=bool, default=False, help='do you want to align the DICs first?')
    args = parser.parse_args()
    main(args)
