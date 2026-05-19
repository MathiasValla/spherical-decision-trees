# Reproducing the Spherical Decision Trees Article Results

This repository contains the prototype implementation, benchmark scripts, raw
result tables, and paper assets for the article "Spherical splits for decision
trees and random forests."

Repository: <https://github.com/MathiasValla/spherical-decision-trees>

## What Is Already Included

- Manuscript source: `paper/spherical_trees_letter.tex`
- Compiled manuscript: `paper/spherical_trees_letter.pdf`
- Article Table 1 source:
  `benchmarks_nonasv/results/spherical_pmlb_target_radial_pruned_sample1000_full_model_summary.csv`
- Per-dataset benchmark means:
  `benchmarks_nonasv/results/spherical_pmlb_target_radial_pruned_sample1000_full_summary.csv`
- Per-fold raw benchmark rows:
  `benchmarks_nonasv/results/spherical_pmlb_target_radial_pruned_sample1000_full.csv`
- Skipped/error datasets:
  `benchmarks_nonasv/results/spherical_pmlb_target_radial_pruned_sample1000_full_errors.csv`
- Figure 1:
  `paper/Figure_1_axis_oblique_spherical.png`
- Figure 2:
  `paper/Figure_2_score_time_tradeoff.png`
- Graphical abstract:
  `paper/graphical_abstract.pdf`, `paper/graphical_abstract.tiff`, and
  `paper/Graphical_Abstract.png`

If you only need to retrieve the reported results, clone the repository and use
the CSV files listed above. The benchmark CSV files are intentionally stored in
plain text so that the article numbers can be checked without rerunning the
full benchmark.

## Environment

The benchmark was developed against the editable package in this repository.
One reproducible setup is:

```bash
git clone https://github.com/MathiasValla/spherical-decision-trees.git
cd spherical-decision-trees
uv venv .venv
uv pip install meson-python ninja
uv pip install -e . --no-build-isolation
uv pip install pandas pmlb matplotlib tabulate "numpy<2" "scikit-learn<1.8"
```

If `uv` is not available, create a Python environment manually and install the
same dependencies with `pip`.

## Regenerating the Full PMLB Benchmark

The article benchmark compares pruned single trees and unpruned forests with
axis-aligned, oblique, and spherical split geometries. Spherical trees use 500
center candidates per node, all data-induced radii for each center, and the
`target_radial` center strategy.

The full benchmark is computationally expensive because spherical split search
uses 500 centers and all data-induced radii per node. On a laptop it may run for
many hours; readers who only need the article results should use the stored CSV
files listed above.

```bash
.venv/bin/python benchmarks_nonasv/bench_spherical_pmlb_full.py \
  --output benchmarks_nonasv/results/spherical_pmlb_target_radial_pruned_sample1000_full.csv \
  --cv 3 \
  --n-estimators 50 \
  --n-center-candidates 500 \
  --radius-candidates none \
  --center-strategy target_radial \
  --prune-single-trees \
  --max-samples-per-dataset 1000 \
  --dataset-timeout 1800 \
  --model-timeout 900 \
  --verbose
```

The script skips datasets that lead to fetch, preprocessing, fitting, or timeout
errors and writes them to the matching `_errors.csv` file. It also writes:

- `_summary.csv`: per-dataset, per-model cross-validation means.
- `_model_summary.csv`: model-level means used in article Table 1.
- `.md`: a readable benchmark summary.

To regenerate those summaries from an existing raw CSV without refitting:

```bash
.venv/bin/python benchmarks_nonasv/bench_spherical_pmlb_full.py \
  --output benchmarks_nonasv/results/spherical_pmlb_target_radial_pruned_sample1000_full.csv \
  --n-center-candidates 500 \
  --radius-candidates none \
  --center-strategy target_radial \
  --prune-single-trees \
  --max-samples-per-dataset 1000 \
  --summarize-only
```

## Regenerating Article Figures

Figure 1 compares axis-aligned, oblique, and spherical frontiers on two-feature
examples:

```bash
.venv/bin/python paper/make_prl_comparison_figure.py
```

Figure 2 plots the score/time trade-off from the benchmark summary:

```bash
.venv/bin/python paper/make_prl_performance_tradeoff_figure.py
```

The graphical abstract can be regenerated with:

```bash
.venv/bin/python paper/make_prl_graphical_abstract.py
```

This writes a PDF and TIFF submission asset, plus a PNG preview, at 1328 by 531
pixels (width by height). The aspect ratio satisfies Elsevier's graphical
abstract minimum of 531 by 1328 pixels (height by width) or proportionally more.

The manuscript PDF can then be rebuilt from the `paper` directory:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error spherical_trees_letter.tex
pdflatex -interaction=nonstopmode -halt-on-error spherical_trees_letter.tex
```

## Additional Exploratory Analyses

The repository also keeps the exploratory analyses that informed the manuscript
but are not all shown in the final paper:

```bash
.venv/bin/python benchmarks_nonasv/analyze_spherical_pmlb_regimes.py
.venv/bin/python benchmarks_nonasv/analyze_spherical_heuristics.py
.venv/bin/python benchmarks_nonasv/plot_spherical_2d_partitions.py
```

Their outputs are stored under `benchmarks_nonasv/results/` and
`benchmarks_nonasv/results/figures/`.

## Article Number Crosswalk

- Table 1 uses
  `benchmarks_nonasv/results/spherical_pmlb_target_radial_pruned_sample1000_full_model_summary.csv`.
- Figure 1 is generated by `paper/make_prl_comparison_figure.py`.
- Figure 2 is generated by `paper/make_prl_performance_tradeoff_figure.py`.
- The manuscript uses the repository URL
  <https://github.com/MathiasValla/spherical-decision-trees> in the experiments
  section and in the data/code availability statement.
