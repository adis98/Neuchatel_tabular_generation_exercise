import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def precompute_cat_DICs(sigmas, cat_dims, timesteps, n_points=200):
    cat_dims = np.array(cat_dims)
    DICs = np.zeros((timesteps, len(cat_dims)))
    c = np.linspace(-8, 8, n_points)
    pdf_c = norm.pdf(c)
    for t in range(timesteps):
        cdf_vals = norm.cdf(c + (1 / (sigmas[t] + 1e-8)))
        integrand_vals = (cdf_vals[:, None] ** (cat_dims[None, :] - 1)) * pdf_c[:, None]
        result = np.trapezoid(integrand_vals, c, axis=0)
        DICs[t] = 1 - result
    return DICs


def match_DICs(reference_DIC, sigmas_base, cardinalities, timesteps):
    corrected_sigmas = np.zeros((timesteps, len(cardinalities)))
    ref_ratio = reference_DIC[:, 0] / reference_DIC[-1, 0]  # (timesteps,)
    for i, k in enumerate(cardinalities):
        # DIC at the shared final sigma for this cardinality
        dic_final = precompute_cat_DICs(np.array([sigmas_base[-1]]), [k], 1)[0, 0]
        # target raw DIC values for this cardinality
        target = ref_ratio * dic_final  # (timesteps,)
        big = np.full(timesteps, 100.0)
        less = np.zeros(timesteps)
        sigmas = sigmas_base.copy()
        for _ in range(20):
            DICs = precompute_cat_DICs(sigmas, [k], timesteps)[:, 0]
            gt = DICs > target
            lt = DICs < target
            big = np.where(gt, sigmas, big)
            less = np.where(lt, sigmas, less)
            sigmas = (big + less) / 2

        # enforce shared start/end exactly
        sigmas[0] = sigmas_base[0]
        sigmas[-1] = sigmas_base[-1]
        corrected_sigmas[:, i] = sigmas
    return corrected_sigmas


def precompute_cat_DICs_fast(sigmas, cat_dims, n_points=200):
    cat_dims = np.array(cat_dims)
    c = np.linspace(-8, 8, n_points)          # (n_points,)
    pdf_c = norm.pdf(c)                        # (n_points,)

    cdf_vals = norm.cdf(
        c[None, :] + (1 / (sigmas[:, None] + 1e-8))
    )  # → (timesteps, n_points)

    integrand_vals = (
        cdf_vals[:, :, None] ** (cat_dims[None, None, :] - 1)
    ) * pdf_c[None, :, None]                  # → (timesteps, n_points, n_cats)

    result = np.trapezoid(integrand_vals, c, axis=1)  # → (timesteps, n_cats)

    return 1 - result


def match_DICs_fast(reference_DIC, sigmas_base, cardinalities):
    cardinalities = np.array(cardinalities)                   # (n_cats,)
    ref_ratio = reference_DIC[:, 0] / reference_DIC[-1, 0]   # (timesteps,)

    # dic_final for all cardinalities at once: shape (1, n_cats) → (n_cats,)
    dic_final = precompute_cat_DICs_fast(
        np.array([sigmas_base[-1]]), cardinalities
    )[0]                                                       # (n_cats,)

    # target: (timesteps, n_cats)
    target = ref_ratio[:, None] * dic_final[None, :]

    # Binary search state, all cardinalities in parallel
    # sigmas: (timesteps, n_cats) — each column starts as sigmas_base
    sigmas = np.tile(sigmas_base[:, None], (1, len(cardinalities)))  # (timesteps, n_cats)
    big = np.full((len(sigmas_base), len(cardinalities)), 100.0)
    less = np.zeros((len(sigmas_base), len(cardinalities)))

    for _ in range(20):
        # Evaluate DICs for every (timestep, cardinality) simultaneously
        # Flatten cardinality columns → run one precompute call per iteration
        DICs = np.stack([
            precompute_cat_DICs_fast(sigmas[:, i], np.array([k]))[:, 0]
            for i, k in enumerate(cardinalities)
        ], axis=1)  # (timesteps, n_cats)

        big  = np.where(DICs > target, sigmas, big)
        less = np.where(DICs < target, sigmas, less)
        sigmas = (big + less) / 2

    # Enforce shared start/end across all cardinalities
    sigmas[0, :] = sigmas_base[0]
    sigmas[-1, :] = sigmas_base[-1]

    return sigmas  # (timesteps, n_cats)


if __name__ == "__main__":
    timesteps = 500
    sigmas = np.linspace(0, 100, timesteps)
    cardinalities = np.array([2, 9, 16, 42])
    # sigmas_expanded = np.repeat(sigmas[:, None], len(cardinalities), axis=1)
    # DICs = precompute_cat_DICs(sigmas, cardinalities, timesteps)
    DICs_fast = precompute_cat_DICs_fast(sigmas, cardinalities)

    reference_DIC = precompute_cat_DICs_fast(sigmas, [2])

    corrected_sigmas = match_DICs_fast(reference_DIC, sigmas, cardinalities)
    # corrected_sigmas_slow = match_DICs(reference_DIC, sigmas, cardinalities, timesteps)

    new_DICs = np.zeros((timesteps, len(cardinalities)))

    for i in range(len(cardinalities)):
        k = cardinalities[i]
        new_DIC = precompute_cat_DICs_fast(corrected_sigmas[:, i], [cardinalities[i]])
        new_DICs[:, i] = new_DIC[:, 0]
    # plt.plot(DICs[:, 0]/DICs[-1, 0], label='k=2')
    # plt.plot(DICs[:, 1]/DICs[-1, 1], label='k=9')
    # plt.plot(DICs[:, 2]/DICs[-1, 2], label='k=16')
    # plt.plot(DICs[:, 3]/DICs[-1, 3], label='k=42')
    # plt.legend()
    # plt.title('Original DICs')
    # plt.show()
    plt.plot(new_DICs[:, 0] / new_DICs[-1, 0], label='k=2')
    plt.plot(new_DICs[:, 1] / new_DICs[-1, 1], label='k=9')
    plt.plot(new_DICs[:, 2] / new_DICs[-1, 2], label='k=16')
    plt.plot(new_DICs[:, 3] / new_DICs[-1, 3], label='k=42')
    plt.legend()
    plt.title('New DICs')
    plt.show()
