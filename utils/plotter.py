import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_distributions(real_df, generated_df, num_cols, cat_cols, ncols=3, aligned=False):
    all_cols = num_cols + cat_cols
    nrows = -(-len(all_cols) // ncols)  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(all_cols):
        ax = axes[i]
        if col in num_cols:
            real_df[col].plot.kde(ax=ax, label="Real", color="steelblue", linewidth=2)
            generated_df[col].plot.kde(ax=ax, label="Generated", color="tomato", linewidth=2, linestyle="--")
            ax.set_ylabel("Density")
        else:
            real_counts = real_df[col].value_counts(normalize=True).sort_index()
            gen_counts = generated_df[col].value_counts(normalize=True).sort_index()
            all_cats = real_counts.index.union(gen_counts.index)
            real_counts = real_counts.reindex(all_cats, fill_value=0)
            gen_counts = gen_counts.reindex(all_cats, fill_value=0)
            x = range(len(all_cats))
            width = 0.4
            ax.bar([p - width / 2 for p in x], real_counts.values, width=width, label="Real", color="steelblue",
                   alpha=0.8)
            ax.bar([p + width / 2 for p in x], gen_counts.values, width=width, label="Generated", color="tomato",
                   alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Proportion")

        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("")

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Real vs Generated Distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"distribution_plots_{aligned}.png", bbox_inches="tight", dpi=150)
    print(f"saved to distribution_plots_{aligned}.png")