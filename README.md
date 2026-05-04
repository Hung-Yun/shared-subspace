# Distinct tasks engage a shared neural subspace

This repository contains analysis code for examining shared neural subspaces across cognitive tasks using intracranial (sEEG) recordings from human epilepsy patients. 

The code is designed to reproduce the core analyses from our study. Data are **not included** due to privacy restrictions.

## Reproducing Analyses
The main analysis notebook is [`analysis.ipynb`](analysis.ipynb). You can run it step by step in Jupyter or VSCode. Dependencies are listed at the top of the notebook.

```
conda env create -f environment.yml
conda activate subspace
pip install -r requirements.txt
```