"""Check completeness and variance conservation for the generated results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--results','--data-dir',default=str(Path(__file__).resolve().parents[2]/'data/subspace_simulation'));args=p.parse_args();root=Path(args.results)
tables, provenance = root/'tables', root/'provenance'
df=pd.read_csv(tables/'results.csv');diag=pd.read_csv(tables/'fit_diagnostics.csv');pairs=pd.read_csv(tables/'pairs.csv')
config=json.loads((provenance/'config.json').read_text())
n_pairs=len(pairs)
n_repeats=int(config['replicates'])
assert n_pairs > 0
assert pairs['T'].eq(config['samples']).all()
assert len(diag)==n_pairs*n_repeats*3*2
assert diag.converged.all(),diag[~diag.converged]
assert len(df)==n_pairs*n_repeats*3*2*6*3
assert df.fraction.between(-1e-10,1+1e-10).all()
key=['pair','replicate','case','session','method']
w=df.pivot(index=key,columns='category',values='fraction')
np.testing.assert_allclose(w.more+w.less,1,atol=2e-12)
text=f'PASS: {len(pairs)} session pairs; {len(diag)} converged FA fits; {len(df)} variance fractions.\nAll fractions in [0,1]; more+less=1 for every task/repeat/case/method.\n'
(provenance/'validation.txt').write_text(text)
print(text)
