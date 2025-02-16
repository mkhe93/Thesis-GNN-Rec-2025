# Content

Further explanations can be found in the corresponding directories and notebooks.

- **`DataPreparation`**: Creates dataset splits and calculates traditional & topological metrics.
  - **`utils`**: Stores all functions used within the notebooks.
  - **[`DataPreparation.ipynb`](DataPreparation/DataPreparation.ipynb)**: Loads the raw data and generates splits.
  - **[`DatasetCharacteristics.ipynb`](DataPreparation/DatasetCharacteristics.ipynb)**: Computes dataset traditional & topological characteristics and generates plots.

- **`HyperParameterTuning`**: Tunes and evaluates all model hyperparameters on the `mids-100000-1` split.
  - **`parameter_fixed`**: Contains the fixed parameters used during the hyperparameter search.
  - **`parameter_space`**: Defines the search spaces for the hyperparameters.
  - **`results`**: Stores the results of each model's hyperparameter search.
  - **`utils`**: Stores all functions used within the notebooks.
  - **[`CreateHyperResultAssets.py`](HyperParameterTuning/CreateHyperResultAssets.py)**: Generates the summary file used in `HyperSearchEvaluation.ipynb`.
  - **[`HyperSearchEvaluation.ipynb`](HyperParameterTuning/HyperSearchEvaluation.ipynb)**: Plots hyperparameters against a target metric (e.g., **NDCG@10**).
  - **[`run_hyper_tuning.py`](HyperParameterTuning/run_hyper_tuning.py)**: Executes the hyperparameter search for a given model.

- **`EpochTuning`**: Determines the optimal number of epochs using 10 randomly drawn datasets.
  - **`EpochEvaluation`**: Stores the log files generated from "epoch tuning" recorded by RecBole.
  - **`utils`**: Stores all functions used within the notebooks.
  - **[`EfficiencyEvaluation.ipynb`](EpochTuning/EfficiencyEvaluation.ipynb)**: Plots the number of epochs against training loss and **NDCG@10**.
  - **[`run_epoch_tuning.py`](EpochTuning/run_epoch_tuning.py)**: Performs "epoch tuning" on 10 randomly drawn data splits.

- **`TestRuns`**: Conducts tests using random order and temporal order splits.
  - **`results`**: Stores the results of each model's test run (RO & TO).
  - **[`run_model_tests.py`](TestRuns/run_model_tests.py)**: Executes the test runs.

- **`Evaluation`**: Builds evaluation files, conducts evaluations, and performs significance tests.
  - **`RO`**: Stores the final `EvaluationData.parquet` for the **RO** split used in `Evaluation.ipynb`.
  - **`TO`**: Stores the final `EvaluationData.parquet` for the **TO** split used in `Evaluation.ipynb`.
  - **`utils`**: Stores all functions used within the notebooks.
  - **[`BuildEvaluationAssets.ipynb`](Evaluation/BuildEvaluationAssets.ipynb)**: Gathers all necessary data to build the `EvaluationData.parquet` files.
  - **[`Evaluation.ipynb`](Evaluation/Evaluation.ipynb)**: Generates evaluation plots and performs significance analysis using [statsmodels](https://www.statsmodels.org/stable/index.html).

- **`AdditionalMaterial`**: Contains additional plots referenced in the thesis (see [README.md](AdditionalMaterial/README.md)).

- **`assets`**: Stores generated plots and statistical outputs.
- **`config`**: Stores `config_files`, constants such as **Colors** and **Paths**, and methods used in many other directories.
