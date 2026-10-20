# Gallstone disease classification: feature-group analysis

Code for *Analyzing the Contribution of Bioimpedance and Laboratory Feature Groups to Machine Learning-Based Gallstone Disease Classification*.

The study compares four predefined feature scenarios using Logistic Regression, SVM, Random Forest, and XGBoost, with GridSearch and Sand Cat Swarm Optimization (SCSO). All model-selection procedures are restricted to the training partition.

## Data

The **Gallstone Dataset** (319 observations, 38 predictors) is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1150/gallstone-1). Download the original spreadsheet and save it as `data/gallstone.xlsx`. Preserve its source column names and row order. The dataset is not redistributed in this repository.

The original target labels are `0` for gallstone disease and `1` for controls; all notebooks recode gallstone disease to the positive class (`1`).

## Setup

The reference environment uses Python **3.13.5**. From the repository root:

```bash
python -m venv .venv
# Activate the environment for your platform.
python -m pip install -r requirements.txt
jupyter notebook
```

Run the numbered notebooks in the `notebooks/` directory. They also detect the repository root when a kernel starts in `notebooks/`. Output files are written to `results/`, which is not version-controlled.

## Notebooks

| Notebook | Analysis | Output |
| --- | --- | --- |
| `01_primary_experiment.ipynb` | 32 primary classifier–scenario–optimizer configurations | `results/gallstone_results_bundle.xlsx` and detailed exports |
| `02_repeated_partitions.ipynb` | Five independent stratified train/test splits | `results/repeated_partitions.xlsx` |
| `03_scso_multiseed.ipynb` | Five SCSO seeds with fixed data split | `results/scso_multiseed.xlsx` |
| `04_correlation_preprocessing.ipynb` | Correlation, VIF, coefficient and preprocessing sensitivity | `results/correlation_preprocessing.xlsx` |
| `05_elastic_net.ipynb` | Elastic Net versus L2 Logistic Regression | `results/elastic_net.xlsx` |
| `06_clinical_evaluation.ipynb` | Clinical metrics, paired bootstrap, SHAP, calibration and DCA | `results/clinical_evaluation.xlsx` |

Run notebook 01 before notebook 06: the latter uses the saved predictions and optimal hyperparameters from the primary experiment. Other supplementary notebooks require only the source data. The SCSO multi-seed and repeated-partition experiments can take considerably longer than the other analyses.

To regenerate manuscript Figures 2–4 after running the relevant notebooks, use `python scripts/plot_figures.py`. This script reads the generated workbooks; it does not retrain models.

## Experimental settings

- Primary train/test split: stratified 80:20, seed **42** (255 training and 64 test observations).
- Primary validation: shuffled, stratified five-fold cross-validation, seed **42**.
- Fixed feature counts: A (11), B (25), C (24), D (38).
- GridSearch and SCSO: **100** fitness evaluations per model and scenario, optimizing mean cross-validation accuracy.
- SCSO population: **10** candidates, followed by **nine** population updates.
- Repeated train/test partitions: seeds **42, 52, 62, 72, 82**.
- Repeated SCSO optimization: seeds **42, 52, 62, 72, 82**, keeping the primary data partition and folds fixed.
- Paired stratified bootstrap: **4,000** resamples, RNG seed **20261017**.

Split indices are generated deterministically from the stated settings and included in the generated run outputs. No precomputed split JSON files are needed.

## Software

`requirements.txt` records the **reference environment used to test the cleaned clinical-analysis notebook and figure script** (Python 3.13.5). The original clinical-analysis workbook independently records Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, and scikit-learn 1.8.0. Complete package versions were not retained for every original modeling run, so this reference environment should not be presented as an exact historical lock for all experiments.

## Reproducibility and interpretation

Bootstrap intervals condition on fitted models and the held-out partition; repeated split evaluation assesses internal stability, not external validation. Clinical calibration and decision-curve analyses are exploratory.

The dataset, intermediate patient-level predictions, and result workbooks are intentionally excluded from the repository. If this work is archived publicly, cite the source dataset and verify its applicable reuse terms. Add a software license only after the authors have selected its terms.
