"""Numerical checks of covariance compression, geometry, and variance summaries."""
import numpy as np
from sklearn.decomposition import FactorAnalysis
from threadpoolctl import threadpool_limits
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.subspace_simulation.simulate_fixed_rank import fit_covariance, random_bases, principal_bases, fractions, sample_data

with threadpool_limits(limits=1):
    rng = np.random.default_rng(42)
    n,d,t=9,4,1200
    x,y,z=random_bases(n,d,rng)
    a,b=np.column_stack([z,x]), np.column_stack([z,y])
    np.testing.assert_allclose(a.T@a,np.eye(d),atol=1e-12)
    np.testing.assert_allclose(b.T@b,np.eye(d),atol=1e-12)
    pa,pb,angles=principal_bases(a,b)
    assert angles[:d//2].max()<2e-6
    v=np.array([5,3,1,.4]); psi=np.linspace(.4,1,n)
    shared=(a*v)@a.T
    f=fractions(shared,a,pa)
    np.testing.assert_allclose([f['top'],f['more'],f['less']], [8/9.4,8/9.4,1.4/9.4],atol=1e-12)
    low=np.column_stack([x,z]); pl,_,_=principal_bases(low,np.column_stack([y,z]))
    fl=fractions((low*v)@low.T,low,pl)
    np.testing.assert_allclose(fl['more'],1.4/9.4,atol=1e-12)
    data=sample_data(a,v,psi,t,rng); data-=data.mean(0)
    cov=data.T@data/t
    compressed,_,ok=fit_covariance(cov,d,t)
    full=FactorAnalysis(n_components=d,svd_method='lapack',tol=1e-6,max_iter=100000).fit(data)
    error=np.max(np.abs(compressed.get_covariance()-full.get_covariance()))
    np.testing.assert_allclose(compressed.get_covariance(),full.get_covariance(),atol=2e-8,rtol=2e-8)
    assert ok
    optimized,_,opt_ok=fit_covariance(cov,d,t,warm_start_iterations=10)
    assert opt_ok and optimized.solver_=='sklearn+profile_lbfgs'
    opt_error=np.max(np.abs(optimized.get_covariance()-full.get_covariance()))
    np.testing.assert_allclose(optimized.get_covariance(),full.get_covariance(),atol=1e-3,rtol=1e-3)
    # Principal-direction calculation is invariant to arbitrary basis signs.
    signs=np.array([-1,1,-1,1]); ps,_,_=principal_bases(a*signs,b)
    np.testing.assert_allclose(fractions(shared,a*signs,ps)['more'],f['more'],atol=1e-12)
    print(f'PASS: geometry, known variance fractions, sign invariance, full-data vs compact FA (max covariance difference {error:.2g}; iterations {full.n_iter_}/{compressed.n_iter_}).')
    print(f'PASS: profile-likelihood fallback agrees with full-data FA (max covariance difference {opt_error:.2g}).')
