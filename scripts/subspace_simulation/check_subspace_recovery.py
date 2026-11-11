"""Audit common-unit PAGDVE outputs in the separate data/models layout."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning

repo=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=Path,default=repo/'data/subspace_simulation')
args=parser.parse_args()
out=args.data_dir.resolve()
tables,provenance=out/'tables',out/'provenance'
sys.path.insert(0,str(repo))
from utils import get_paired_df
from decomposition import FAResults
from scripts.subspace_alignment import pagdve, sha256
from scripts.subspace_simulation.generate_activity import generate_activity

cfg=json.loads((provenance/'config.json').read_text())
if not cfg.get('common_unit_cache'):
    raise SystemExit('This verifier requires common-unit FA inputs. Use check_fixed_rank_results.py for Jay\'s fixed-D results.')
model_dir=(out/cfg['directories']['models']).resolve()
cache=Path(cfg['common_unit_cache'])
if not cache.is_absolute(): cache=repo/cache
assert sha256(cache)==cfg['input_sha256']['common_unit_cache']
rows=get_paired_df(pd.read_hdf(cache,'all'))
by_pair={tuple(sorted([r.task_1.session_key,r.task_2.session_key])):r for r in rows.itertuples()}
pairs=pd.read_csv(tables/'pairs.csv')
comp=pd.read_csv(tables/'comparisons.csv')
ve=pd.read_csv(tables/'results.csv')
fits=pd.read_csv(tables/'fit_diagnostics.csv')
expected={(p,rep,c) for p in pairs.pair for rep in range(cfg['replicates']) for c in (1,2,3)}
assert len(comp)==len(expected)
assert set(comp[['pair','replicate','case']].itertuples(index=False,name=None))==expected
assert not comp.duplicated(['pair','replicate','case']).any()
assert len(fits)==2*len(expected) and fits.converged.all()
assert not fits.duplicated(['pair','replicate','case','session']).any()
assert len(ve)==6*len(expected) and not ve.duplicated(['pair','replicate','case','session','category']).any()
assert pairs['T'].eq(cfg['samples']).all()
common_metadata=json.loads((out/'inputs/common_units.json').read_text())
model_arrays={}
fit_errors=[]
with threadpool_limits(limits=1):
    for pair in pairs.itertuples():
        row=by_pair[tuple(sorted([pair.session_a,pair.session_b]))]
        sessions={t.session_key:t for t in (row.task_1,row.task_2)}
        channels=sorted(row.common)
        for side,meta in enumerate(common_metadata[pair.pair],1):
            task=sessions[meta['session']]
            positions={name:i for i,name in enumerate(task.df_chan[meta['channel_field']])}
            ix=np.array([positions[c] for c in channels])
            assert meta['channels']==[str(c) for c in channels]
            assert meta['source_indices']==ix.tolist()
            path=model_dir/meta['model_path']
            with np.load(path,allow_pickle=False) as saved, np.load(out/meta['input_path'],allow_pickle=False) as inputs:
                Y=task.spikes_per_bin[:,ix]
                np.testing.assert_array_equal(inputs['Y'],Y)
                Z=np.nan_to_num((Y-Y.mean(0))/Y.std(0),nan=0.,posinf=0.,neginf=0.)
                np.testing.assert_array_equal(inputs['Y_norm'],Z)
                assert hashlib.sha256(Y.tobytes()).hexdigest()==meta['input_sha256']
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    refit=FAResults().fa_fit(Z,max_iter=10000,tol=1e-4,svd_method='lapack',shared_var_thresh=.95)
                assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                np.testing.assert_allclose(saved['components'],refit.fa.components_,atol=1e-10,rtol=1e-8)
                np.testing.assert_allclose(saved['private_variances'],refit.fa.noise_variance_,atol=1e-10,rtol=1e-8)
                spectrum=np.linalg.svd(refit.fa.components_,compute_uv=False)**2
                np.testing.assert_allclose(saved['shared_spectrum'],spectrum,atol=1e-10,rtol=1e-8)
                fit_errors.append(float(np.max(abs(saved['components']-refit.fa.components_))))
                assert refit.d_shared==int(saved['retained_rank'])
                model_arrays[str(path.relative_to(model_dir))]=(saved['shared_spectrum'].copy(),saved['private_variances'].copy(),len(channels))
        assert pair.D==2*(min(pair.common_M1,pair.common_M2)//2)
        assert pair.N==len(channels)
    assert sha256(model_dir/'generating_params.pkl')==cfg['generating_params_sha256']
    with (model_dir/'generating_params.pkl').open('rb') as f:
        generating=pickle.load(f)
    records=[ab for group in generating.values() for ab in group]
    assert len(records)==len(expected)
    for a,b in records:
        assert a['common_channels']==b['common_channels']
        for p in (a,b):
            spectrum,private,n=model_arrays[p['common_fit_path']]
            assert p['parameter_source']=='common_unit_fa'
            assert p['T']==cfg['samples'] and p['N']==n
            np.testing.assert_array_equal(p['private_variances'],private)
            np.testing.assert_array_equal(p['shared_variances'],spectrum[:p['D']])
            np.testing.assert_allclose(p['basis'].T@p['basis'],np.eye(p['D']),atol=1e-12)
    saved_results=list((out/'comparisons').rglob('*.npz'))
    assert len(saved_results)==len(expected)
    for path in saved_results:
        with np.load(path,allow_pickle=False) as r:
            assert r['data_pa'].shape==(int(r['d']),) and r['data_ve'].shape==(8,)
            assert int(r['K'])==int(r['d'])//2
            np.testing.assert_allclose(np.diag(r['C1']),1,atol=1e-12)
            np.testing.assert_allclose(np.diag(r['C2']),1,atol=1e-12)
            meta=json.loads(str(r['metadata']))
            with np.load(model_dir/meta['model_path'],allow_pickle=False) as bases:
                for side in (1,2):
                    basis=bases[f'L{side}']
                    assert basis.shape[1]==r['ranks'][side-1]
                    np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-12)
            assert 'L1' not in r.files and 'L2' not in r.files
    replay=[]
    for case in (1,2,3):
        pair=pairs.pair.iloc[(case-1)%len(pairs)]
        a,b=next((a,b) for a,b in records if a['pair_id']==pair and a['replicate']==0 and a['case']==case)
        Y1,Y2=generate_activity(a),generate_activity(b)
        result=pagdve(Y1,Y2,pair_id=pair,**cfg['analysis'])
        with np.load(out/'comparisons'/pair/f'case_{case}'/'repeat_000.npz',allow_pickle=False) as saved:
            meta=json.loads(str(saved['metadata']))
            assert meta['input_sha256']==[hashlib.sha256(Y.tobytes()).hexdigest() for Y in (Y1,Y2)]
            for name in ('data_pa','data_gd','data_ve','denominators','ranks'):
                np.testing.assert_array_equal(saved[name],result[name])
        replay.append(dict(pair=pair,case=case,replicate=0,exact_replay=True))
audit=dict(status='pass',comparisons=len(comp),simulated_fa_fits=len(fits),
           common_unit_fits_replayed=len(model_arrays),max_common_refit_component_error=max(fit_errors),actual_common_units_verified=True,
           spectra_and_private_variances_match_common_fits=True,
           all_combinations_present=True,cache_unchanged=True,replay_checks=replay)
(provenance/'verification.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
