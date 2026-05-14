import numpy as np


def Zipf_CDF(n, alpha):
    """Computes Zipf cumulative distribution function.
    Each percentage corresponds to word index in array"""
    zipf_law = np.power(np.arange(1, n + 1), -alpha)
    zeta = np.cumsum(zipf_law)
    return zeta / zeta[-1]


def Zipf_Mandelbrot_CDF(n, alpha, beta=2.7):
    """Computes Zipf-Mandelbrot cumulative distribution function"""
    x = np.arange(1, n + 1) + beta
    mandelbrot_law = np.power(x, -alpha)
    zeta = np.cumsum(mandelbrot_law )
    return zeta / zeta[-1]


def Zipf_Mand_3S_CDF(n, alpha_1=1.16, alpha_2=1.48, alpha_3=1.866,
                     beta=6.9, c1=2041507.88, c2=9105.72, c3=126.287,
                     N1=100, N2=2000):
    """Computes Zipf-Mandelbrot cumulative distribution function in three stages"""

    x = np.arange(1, n + 1) + beta
    v1 = c1 * np.power(x[:N1], -alpha_1)
    v2 = N1 ** alpha_2 * c2 * np.power(x[N1:N2], -alpha_2)
    v3 = N2 ** alpha_3 * c3 * np.power(x[N2:], -alpha_3)

    mandelbrot_law_3S = np.concatenate((v1, v2, v3))
    zeta = np.cumsum(mandelbrot_law_3S)
    return zeta / zeta[-1]


def Zipf_CDF_compressed(n, alpha, n_red=1000):
    """Computes Zipf cumulative distribution function.
    It compresses real long interval n into a smaller n_red
    by conserving relative percentages by intervals
    Each percentage corresponds to word index in array"""

    zipf_law = np.power(np.arange(1, n + 1), -alpha)

    zeta = np.cumsum(zipf_law)
    zeta = zeta / zeta[-1]
    interv_div = np.linspace(1, n, n_red + 1).astype(np.int64)

    # zeta = np.array([zeta[i1-1] for i1 in interv_div[1:]])
    zeta = zeta[interv_div[1:] - 1]

    return zeta


def Zipf_Mand_CDF_compressed(n, alpha, beta=2.7, n_red=1000):
    """Computes Zipf cumulative distribution function.
    It compresses real long interval n into a smaller n_red
    by conserving relative percentages by intervals
    Each percentage corresponds to word index in array"""

    x = np.arange(1, n + 1) + beta
    zipf_mand_law = np.power(x, -alpha)

    zeta = np.cumsum(zipf_mand_law)
    zeta = zeta / zeta[-1]
    interv_div = np.linspace(1, n, n_red + 1).astype(np.int64)

    # zeta = np.array([zeta[i1-1] for i1 in interv_div[1:]])
    zeta = zeta[interv_div[1:] - 1]
    return zeta

# IDEA : to model vocab_size vs age dependency, play both with n and n_red in following function
# Use factor for n, n_red ???
def Zipf_Mand_3S_CDF_comp(n, alpha_1=1.16, alpha_2=1.48, alpha_3=1.866,
                          beta=6.9, c1=2041507.88, c2=9105.72, c3=126.287,
                          N1=100, N2=2000, n_red=1000):
    """Computes Zipf-Mandelbrot cumulative distribution function in three stages
       It compresses real long interval n into a smaller n_red
       by conserving relative percentages by intervals"""

    x = np.arange(1, n + 1) + beta
    v1 = c1 * np.power(x[:N1], -alpha_1)
    v2 = N1 ** alpha_2 * c2 * np.power(x[N1:N2], -alpha_2)
    v3 = N2 ** alpha_3 * c3 * np.power(x[N2:], -alpha_3)

    mandelbrot_law_3S = np.concatenate((v1, v2, v3))
    zeta = np.cumsum(mandelbrot_law_3S)
    zeta = zeta / zeta[-1]

    interv_div = np.linspace(1, n, n_red + 1).astype(np.int64)
    zeta = zeta[interv_div[1:] - 1]
    return zeta


def vocab_ceiling_curve(age_steps, steps_per_year=36, n_min=500, n_max=10000,
                        midpoint_years=8, rate=0.4):
    """Age-dependent ceiling on the raw vocabulary size an agent can sample from.

    Logistic growth in years: from n_min in early childhood to n_max in adulthood.
    Replaces the legacy externally-precomputed age->vocab mapping. Returns an int
    (or int array) suitable as the `n` argument to the Zipf CDF generators.
    """
    age_years = np.asarray(age_steps) / steps_per_year
    n = n_min + (n_max - n_min) / (1 + np.exp(-rate * (age_years - midpoint_years)))
    return np.rint(n).astype(np.int64)


def randZipf(zipf_cum_distr, numSamples):
    """fast computation of array of Zipf samples with dim = numSamples
    It needs Zipf CDF as input"""
    unif_random_array = np.random.random(numSamples)
    return np.searchsorted(zipf_cum_distr, unif_random_array)


def randZipf_dim(n, alpha, numSamples, dim = 10):
    """ TODO: same as randZipf but for higher dims
    To be completed if needed """
    tmp = np.power( np.arange(1, n+1), -alpha )
    #tmp[0] = 0.0
    zeta = np.cumsum(tmp)
    # Store the translation map:
    distMap = zeta / zeta[-1]
    # distMap = np.repeat(distMap[:, np.newaxis], dim, 1)
    # Generate an array of uniform 0-1 pseudo-random values:
    u = np.random.random(dim * numSamples)
    # bisect them with distMap
    out_array = (np.searchsorted(distMap, u) - 1).reshape(100,dim)
    return out_array