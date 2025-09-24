#%% imports

import numpy as np
import scipy
from numpy.random import default_rng
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score

#%% helper functions

def example_data(n_samples, n_features, rank, sigma=1.0, seed=None):
    """
    Generate example data for Factor Analysis.
    n_samples: number of samples (observations)
    n_features: number of features (dimensions)
    rank: rank of the latent space (number of latent dimensions)
    sigma: scale of the heteroscedastic noise
    seed: random seed for reproducibility
    Returns a (n_samples x n_features) matrix of observations.
    The observations are generated as a linear combination of latent variables with added heteroscedastic noise.
    """
    np.random.seed(seed)
    U, _, _ = scipy.linalg.svd(np.random.randn(n_features, n_features))
    L = U[:,:rank] # (n_features x rank)
    Z = np.random.randn(n_samples, rank) # generate latents: (n_samples x rank)
    X = np.dot(Z, L.T) # observations: (n_samples x n_features)

    # add heteroscedastic noise to observations
    sigmas = sigma * np.random.randn(n_features) + sigma / 2.0
    return X + np.random.randn(n_samples, n_features) * sigmas

def fa_transform(X, fa, orthonormalize=True):
    """
    Transform data X using a fitted Factor Analysis model fa.
    X is a (N x D) matrix of observations, where N is the number of samples and D is the number of features.
    fa is a fitted FactorAnalysis object.
    orthonormalize is a boolean indicating whether to orthonormalize the latent variables.
    
    Returns the transformed latent variables Z, and optionally the orthonormal basis L_tilde.
    If orthonormalize is True: returns Z_tilde and L_tilde, where:
    - Z_tilde is the transformed latent variables in the orthonormal basis.
    - L_tilde is the (DxK) orthonormal basis for the subspace defined by the first K dimensions.
    If orthonormalize is False: returns Z, the posterior mean of the latent variables.
    """
    # get posterior mean of latent variables
    L = fa.components_.T # (D x n_components)
    Ph = fa.noise_variance_ # (D,)
    P = (L.T @ np.linalg.inv(L @ L.T + np.diag(Ph))).T # (D x n_components)
    Z = np.dot(X - fa.mean_, P) # (N x n_components)

    if not orthonormalize:
        return Z
    
    # spikes arise from latents as:
    #   Y = Z @ L.T = (Z @ V @ S) @ U.T
    #       where U.T is an orthonormal basis
    #   so we will define Z_tilde = Z @ V @ S
    #   and L_tilde = U (orthonormal basis for subspace)
    [U,S,V] = np.linalg.svd(L, full_matrices=False)
    Z_tilde = Z @ V @ np.diag(S) # (N x n_components), orthonormalized latent variables
    L_tilde = U # (D x n_components), orthonormal basis for subspace
    return Z_tilde, L_tilde

def fa_crossval_scores(X, n_components, verbose=True):
    """
    Compute cross-validation scores for Factor Analysis with varying numbers of components.
    X is a (N x D) matrix of observations, where N is the number of samples and D is the number of features.
    n_components is a list of numbers of components to use for cross-validation.
    Returns a numpy array of cross-validation scores for each number of components.
    If verbose is True, it will print the cross-validation scores for each number of components.
    """
    fa_scores = []
    for n in n_components:
        fa = FactorAnalysis()
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X)))
        if verbose:
            print(f'CV score for {n} components: {fa_scores[-1]}')
    return np.array(fa_scores)

def fa_fit(X, n_components=None, cv_components=None, shared_var_thresh=0.95, max_iter:int=int(1e6), verbose=False):
    """
    X is a (N x D) matrix of observations, where N is the number of samples and D is the number of features.
    n_components is the number of latent dimensions to fit, or None to use all features.
    cv_components is a list of numbers of components to use for cross-validation, or None to skip CV.
    shared_var_thresh is the proportion of shared variance to use for determining d_shared.
    verbose is a boolean indicating whether to print progress messages.
    Returns a dictionary with the fitted FactorAnalysis object, cross-validation scores, number of components used,
    proportion of shared variance, and the orthonormal basis for the subspace defined by the first d_shared dimensions.
    If verbose is True, it will print the number of shared dimensions found.
    
    The params n_components and cv_components determine how many latent dimensions to fit.
    - If n_components is provided, FA will find that many components directly.
    - If cv_components is provided, the selected number of FA components is based on cross-validation scores.
    - If neither is provided, FA will use all features, and d_shared will be determined by shared_var_thresh.
    """

    # apply cross-validation if needed
    scores = None
    if cv_components is not None:
        # do cross-validation to select number of components
        if n_components is not None:
            raise Exception('Cannot provide both n_components and cv_components.')
        scores = fa_crossval_scores(X, cv_components)
        n_components = cv_components[np.argmax(scores)]
    elif n_components is None:
        n_components = X.shape[1]

    # fit FA
    fa = FactorAnalysis(n_components=n_components, svd_method='lapack', tol=1e-6, max_iter=max_iter)
    fa.fit(X)
    L = fa.components_.T # (D x n_components)

    # get proportion of shared variance and d_shared
    [U,shared_var,V] = np.linalg.svd(L @ L.T)
    prop_shared_var = shared_var / shared_var.sum()
    if n_components is None or n_components == X.shape[1]:
        d_shared = np.where(np.cumsum(prop_shared_var) >= shared_var_thresh)[0][0] + 1
    else:
        d_shared = n_components
    if verbose:
        print(f'{d_shared=}')

    # get shared variance per unit
    Lc = L[:,:d_shared] # (D x d_shared)
    Ph = fa.noise_variance_ # (D,)
    Cov = Lc @ Lc.T + Ph # (D x D)
    shared_var_per_unit = np.diag(Lc @ Lc.T) / np.diag(Cov) # (D,)

    # get orthonormal basis for subspace defined by first d_shared dims
    [Lrot,s,v] = np.linalg.svd(Lc, full_matrices=False)

    return {'fa': fa, 
            'cv_scores': scores, 
            'cv_components': cv_components, 
            'n_components': n_components, 
            'shared_var_per_unit': shared_var_per_unit, 
            'explained_variance_ratio_': prop_shared_var, 
            'd_shared': d_shared, 
            'subspace': Lrot.T,
            'first_d_components': Lc.T}


# %%
