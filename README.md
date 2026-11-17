# Run the example

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run this command from the repository root:

```bash
uv run python example_analysis.py
```

The command prepares the Python environment and runs the analyses on two supplied anonymized sessions. Numerical results appear in the terminal, and seven PDF figures are saved in `examples/results/` and opened in the default PDF viewer when available. No separate environment activation or access to the restricted study data is needed. The first run includes dependency downloads; subsequent runs reuse the environment.

The example covers task decoding (Fig. 2I), factor analysis (Fig. 3B/F), regional comparisons (Fig. 4), principal angles and subspace distance (Fig. 5), proportional variance explained (Fig. 6C), paired firing rates (Fig. S6C), and dimensionality estimates (Fig. S7). Shuffle, shift-control, and dimensionality results are supplied as saved numerical arrays. Expected values and data fields are listed in [examples/README.md](examples/README.md).

# System requirements

Python 3.13 is selected by `.python-version`. Dependencies are defined in [pyproject.toml](pyproject.toml), with resolved versions recorded in [uv.lock](uv.lock). [requirements.txt](requirements.txt) is generated from that lock for computation-script exports. No GPU or other specialized hardware is required. The example has been tested with Python 3.13.11 on macOS 15.7.7 (arm64); other operating systems have not been verified.

# Optional notebook

To view the same analyses with inline figures, run this command from the repository root:

```bash
uv run --extra notebook jupyter lab examples/reviewer_example.ipynb
```

Open [reviewer_example.ipynb](examples/reviewer_example.ipynb), select the Python 3 kernel, and run all cells. Its working directory should be `examples/`.

# Shared-subspace analyses

The six analysis files contain interactive `#%%` cells organized by figure panels. Run interactive cells with the repository root as the working directory. These full-study scripts require the original study data, which are separate from the supplied example sessions.

| File | Analyses |
| --- | --- |
| `main_analysis.py` | General characterization, example traces, and shared variability |
| `subspace_comparison.py` | Subspace comparisons and alignment simulations |
| `regions.py` | Regional contributions and shift-control results |
| `decode.py` | Decoding |
| `tuning.py` | Tuning and responsiveness |
| `sensory_inputs.py` | Sensory-control comparisons, statistics, and energy distributions |

## Helper files

- `data_loader.py` contains session and recording readers.
- `decomposition.py` contains FA fitting.
- `utils.py` contains shared analysis and plotting helpers, including neural-bin timestamp regeneration for sensory-input processing.

## Computation scripts

- `scripts/subspace_alignment.py` contains calculations of principal angles, geodesic distances, and variance explained.
- `scripts/shift_controls.py` contains regional shift controls with fixed-rank FA fitting.
- `scripts/subspace_simulation/` contains scripts for generating activity, simulating subspace alignment, and validating results.
- `scripts/sensory_inputs/` contains scripts for synchronizing and stitching videos, extracting full-frame optical-flow motion energy, and calculating acoustic RMS in neural time bins. These measurements feed `sensory_inputs.py` for Fig S11–S12. See the [workflow guide](scripts/sensory_inputs/README.md) for commands and environment requirements, and the [processing record](scripts/sensory_inputs/README_process_motion.md) for processing choices and retained outputs.

# Figure index

## Script → figures

| Script | Figures plotted |
| --- | --- |
| [main_analysis.py](main_analysis.py) | Fig 2D–F + Fig S4; Fig 3B; Fig 3C–E; Fig 3F; Fig 3G; Fig S1; Fig S2; Fig S6; Fig S7; Fig S9 |
| [subspace_comparison.py](subspace_comparison.py) | Fig 5; Fig 6; Fig S10; Fig S13 |
| [regions.py](regions.py) | Fig 4; Fig S8D |
| [decode.py](decode.py) | Fig 2G; Fig 2I; Fig 3H |
| [tuning.py](tuning.py) | Fig 2H; Fig 2J; Fig S5 |
| [sensory_inputs.py](sensory_inputs.py) | Fig S11; Fig S12 |

## Figure panels → script

| Figure panels | Plotting script |
| --- | --- |
| Fig 2D–F | [main_analysis.py](main_analysis.py) |
| Fig 2G | [decode.py](decode.py) |
| Fig 2H | [tuning.py](tuning.py) |
| Fig 2I | [decode.py](decode.py) |
| Fig 2J | [tuning.py](tuning.py) |
| Fig 3B | [main_analysis.py](main_analysis.py) |
| Fig 3C–E | [main_analysis.py](main_analysis.py) |
| Fig 3F | [main_analysis.py](main_analysis.py) |
| Fig 3G | [main_analysis.py](main_analysis.py) |
| Fig 3H | [decode.py](decode.py) |
| Fig 4 | [regions.py](regions.py) |
| Fig 5 | [subspace_comparison.py](subspace_comparison.py) |
| Fig 6 | [subspace_comparison.py](subspace_comparison.py) |
| Fig S1 | [main_analysis.py](main_analysis.py) |
| Fig S2 | [main_analysis.py](main_analysis.py) |
| Fig S4 | [main_analysis.py](main_analysis.py) |
| Fig S5 | [tuning.py](tuning.py) |
| Fig S6 | [main_analysis.py](main_analysis.py) |
| Fig S7 | [main_analysis.py](main_analysis.py) |
| Fig S8D | [regions.py](regions.py) |
| Fig S9 | [main_analysis.py](main_analysis.py) |
| Fig S10 | [subspace_comparison.py](subspace_comparison.py) |
| Fig S11 | [sensory_inputs.py](sensory_inputs.py) |
| Fig S12 | [sensory_inputs.py](sensory_inputs.py) |
| Fig S13 | [subspace_comparison.py](subspace_comparison.py) |

# License

The custom code and its documentation are licensed under the [MIT License](LICENSE). Third-party components and dependencies retain their original licenses.
Research data are not covered by this software license; access to the restricted study data requires permission from the corresponding author.
