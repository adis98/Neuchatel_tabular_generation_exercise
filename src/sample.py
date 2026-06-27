import torch
import numpy as np
import argparse
import os
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils import Preprocessor
from utils.model import DiffusionMLP
from utils.dic_experimenter import precompute_cat_DICs_fast, match_DICs_fast
from utils.plotter import plot_distributions

warnings.filterwarnings('ignore')
torch.set_num_threads(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prepper = Preprocessor(args.dataname)
    test_df = prepper.df_test
    test_data_encoded = prepper.encodeDfToNp(test_df)
    test_data_standardized = prepper.standardize_np(test_data_encoded)
    x_test = torch.tensor(test_data_standardized, dtype=torch.float32)
    model = DiffusionMLP(in_dim=test_data_standardized.shape[1]).to(device)
    models_dir = f'saved_models/{args.dataname}/'
    if args.alignDIC:
        model_path = os.path.join(models_dir, "model_aligned.pt")
    else:
        model_path = os.path.join(models_dir, "model_unaligned.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    num_dim = len(prepper.num_cols)
    model.eval()
    cat_dims = [len(cats) for cats in prepper.OneHotEncoder.categories_]
    replica_counts = (torch.tensor(np.array(([1] * len(prepper.num_cols)) + cat_dims))).to(device)

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
    print("starting sampling..")
    with torch.no_grad():
        x_t = torch.normal(0, 1, size=x_test.shape, device=device)
        for t in tqdm(range(args.timesteps, 1, -1), desc="Sampling", unit="step"):
            t_normalised = torch.full(size=(x_t.shape[0],), fill_value=(t - 1) / (args.timesteps - 1),
                                      dtype=torch.float32).to(
                device)
            sigmas = torch.tensor(corrected_sigmas[t - 1].reshape(1, -1).repeat(x_t.shape[0], axis=0),
                                  dtype=torch.float32).to(device)
            sigmas_expanded = sigmas.repeat_interleave(replica_counts, dim=1)
            if t == args.timesteps:
                x_t *= sigmas_expanded

            sigmas_prev = torch.tensor(corrected_sigmas[t - 2].reshape(1, -1).repeat(x_t.shape[0], axis=0),
                                       dtype=torch.float32).to(device)
            c_skip = 1 / (1 + sigmas_expanded ** 2)

            outs = model(torch.sqrt(c_skip) * x_t, t_normalised)

            sigmas_prev_expanded = sigmas_prev.repeat_interleave(replica_counts, dim=1)
            c_out = sigmas_expanded / ((1 + sigmas_expanded ** 2).sqrt())
            x0_pred = c_skip[:, :num_dim] * x_t[:, :num_dim] + c_out[:, :num_dim] * outs[:, :num_dim]
            cat_chunks_pred = torch.split(outs[:, num_dim:], cat_dims, dim=-1)
            for chunk in cat_chunks_pred:
                x0_pred = torch.concat((x0_pred, F.softmax(chunk, dim=-1)), dim=-1)
            epsilon = (x_t - x0_pred) / sigmas_expanded
            x_t = x_t + (sigmas_prev_expanded - sigmas_expanded) * epsilon

    generated_np = x_t.cpu().numpy()
    generated_np_destd = prepper.destandardize_np(generated_np)
    generated = prepper.decodeNpToDf(generated_np_destd)
    plot_distributions(test_df, generated, prepper.num_cols, prepper.cat_cols, aligned=args.alignDIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Missing Value Imputation')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--batch_sz', type=int, default=4096)
    parser.add_argument('--timesteps', type=int, default=500, help='Number of diffusion steps.')
    parser.add_argument('--sigma_max', type=float, default=100, help='last variance schedule')
    parser.add_argument('--alignDIC', type=bool, default=False, help='do you want to align the DICs first?')
    args = parser.parse_args()
    main(args)
