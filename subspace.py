import numpy as np

def principal_angles(A, B, return_vectors=False):
    """
    compute the principal angles (degrees) between subspaces A and B, in ascending order
    
    inputs:
        - A: (M, N) array_like
        - B: (M, K) array_like
    returns:
        - angles: ndarray, shape (min(N, K),)

    note: identical to np.rad2deg(scipy.linalg.subspace_angles(A,B)[::-1]) but faster
    """
    # Orthogonalize columns of A and B (QR decomposition)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    
    # Compute the singular values of the overlap matrix
    Ua, S, Ub = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    
    # Compute the principal angles (in radians)
    angles = np.arccos(np.clip(S, -1, 1))  # Clip to avoid numerical issues
    angles = np.rad2deg(angles)

    # Return angles or full dict of angles and vectors
    return angles if not return_vectors else (angles, Ua, Ub)

def rand_principal_angles(A, B, niters, return_vectors=False):
    """
    calculate the principal angles of randomly sampled subspaces
        that match the dimensions of A and B
    """
    results = []
    for _ in range(niters):
        Xrand = np.random.randn(*A.shape)
        Yrand = np.random.randn(*B.shape)
        angles = principal_angles(Xrand, Yrand, return_vectors=return_vectors)
        results.append(angles)
    return np.vstack(results) if not return_vectors else results

def geodesic_distance(A, B):
    """
    Computes the Geodesic distance between two subspaces A and B.
    The distance is sqrt(sum(theta_i^2))) where theta_i are principal angles
    Returns distance in units of degrees
    """
    angles_deg = principal_angles(A, B) # This returns angles in degrees

    if angles_deg is None: # Should not happen with current principal_angles implementation
        return np.nan 
    if not isinstance(angles_deg, np.ndarray) or angles_deg.ndim == 0: # Handles scalar or non-array returns unexpectedly
        # If principal_angles could return a single scalar angle
        if isinstance(angles_deg, (int, float)):
             angles_deg = np.array([angles_deg])
        else: # Unrecognized format
            return np.nan
            
    if angles_deg.size == 0: # If subspaces are such that no angles are defined (e.g. one is zero-dim)
        return 0.0 # Or np.nan, 0.0 implies identical if one is empty, which is debatable

    return np.linalg.norm(angles_deg)
