#!/usr/bin/env python3
"""Run independent session pairs in separate processes, then collect outputs."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.subspace_simulation.simulate_fixed_rank import ROOT, DEFAULT_SAMPLES, common_neuron_counts, pairs


def worker(command):
    p=subprocess.run(command,capture_output=True,text=True)
    return p.returncode,p.stdout,p.stderr


def matching_parameters(metadata, pair_id, samples, common_count):
    """Do not reuse old 10k-sample or differently sized simulations."""
    row = metadata.loc[metadata.pair == pair_id]
    if len(row) != 1 or int(row.iloc[0]['T']) != samples:
        return False
    source = row.iloc[0].get('N_source', 'minimum_model_count')
    if common_count is None:
        return source == 'minimum_model_count'
    return source == 'empirical_common_count' and int(row.iloc[0]['N']) == common_count


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--output',default=str(ROOT / 'data/subspace_simulation'));p.add_argument('--replicates',type=int,default=100);p.add_argument('--jobs',type=int,default=2);p.add_argument('--reuse',default=None)
    p.add_argument('--samples',type=int,default=DEFAULT_SAMPLES)
    p.add_argument('--pair-manifest',default=None,
                   help='Empirical PAGDVE manifest: use its common-neuron counts and included pairs.')
    a=p.parse_args();out=Path(a.output)
    tables, provenance = out/'tables', out/'provenance'
    if (tables/'results.csv').exists():
        raise FileExistsError(f'Preserving existing results in {out}; choose a new data destination.')
    tables.mkdir(parents=True,exist_ok=True);provenance.mkdir(parents=True,exist_ok=True)
    work=out/'comparisons';work.mkdir(exist_ok=True)
    if a.samples < 2: p.error('samples must be at least 2')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');models=joblib.load(a.models)
    pair_list=list(pairs(models));start=time.time();sources=[];commands=[]
    common_counts = common_neuron_counts(a.pair_manifest)
    if common_counts is not None and set(common_counts) - set(pair_list):
        raise ValueError('Empirical pairs are missing from the supplied models.')
    # Original indices keep the same pair seeds when the empirical cohort is selected.
    indexed_pairs = [(i, pair) for i, pair in enumerate(pair_list)
                     if common_counts is None or pair in common_counts]
    reuse_results={}
    if a.reuse:
        old={f:pd.read_csv(Path(a.reuse)/'tables'/f) for f in ['results.csv','pairs.csv','fit_diagnostics.csv']}
        for i,(sa,sb) in indexed_pairs:
            pid=sa+'__'+sb;diag=old['fit_diagnostics.csv'];diag=diag[diag.pair==pid]
            common_count = None if common_counts is None else common_counts[(sa, sb)]
            if (matching_parameters(old['pairs.csv'], pid, a.samples, common_count)
                    and len(diag)==a.replicates*6 and diag.converged.all()):
                reuse_results[i]=pid
    for i,(sa,sb) in indexed_pairs:
        pid = sa+'__'+sb
        common_count = None if common_counts is None else common_counts[(sa, sb)]
        dest=work/f'pair_{i:02d}';dest.mkdir(exist_ok=True);sources.append(dest)
        (dest/'tables').mkdir(exist_ok=True)
        if all((dest/'tables'/f).exists() for f in ['results.csv','pairs.csv','fit_diagnostics.csv']):
            diag=pd.read_csv(dest/'tables/fit_diagnostics.csv')
            meta=pd.read_csv(dest/'tables/pairs.csv')
            if (matching_parameters(meta, pid, a.samples, common_count)
                    and len(diag)==a.replicates*6 and diag.converged.all()
                    and diag.pair.eq(pid).all()):
                continue
        if i in reuse_results:
            pid=reuse_results[i]
            for name,frame in old.items():frame[frame.pair==pid].to_csv(dest/'tables'/name,index=False)
            continue
        command = [sys.executable,str(Path(__file__).with_name('simulate_fixed_rank.py')),'--models',a.models,'--output',str(dest),'--replicates',str(a.replicates),'--pair-index',str(i),'--samples',str(a.samples)]
        if a.pair_manifest is not None:
            command.extend(['--pair-manifest', a.pair_manifest])
        commands.append(command)
    print(f'{len(indexed_pairs)} pairs; reusing {len(reuse_results)} completed pairs; running {len(commands)} with {a.jobs} workers',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures={pool.submit(worker,c):c for c in commands}
        for future in concurrent.futures.as_completed(futures):
            code,stdout,stderr=future.result();command=futures[future]
            if code:raise RuntimeError(f'{command}: {stderr}\n{stdout}')
            print('Finished pair',command[command.index('--pair-index')+1],flush=True)
    for name in ['results.csv','pairs.csv','fit_diagnostics.csv']:
        pd.concat([pd.read_csv(d/'tables'/name) for d in sources],ignore_index=True).to_csv(tables/name,index=False)
    configs = [d/'provenance/config.json' for d in sources]
    if a.reuse:
        configs.append(Path(a.reuse)/'provenance/config.json')
    sample_config=next(path for path in configs if path.exists())
    config=json.loads(sample_config.read_text());config.update(output=a.output,pair_index=-1,jobs=a.jobs,elapsed_seconds=time.time()-start)
    config.update(samples=a.samples,pair_manifest=a.pair_manifest)
    (provenance/'config.json').write_text(json.dumps(config,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(layout_version=2,dataset='fixed_D_simulation',
        config='provenance/config.json'),indent=2));print('All pairs complete',time.time()-start,flush=True)

if __name__=='__main__':main()
