# 🧠 Master Thesis Project: Evaluating Medical AI with Demography-Conditioned Synthetic Test Sets

This repository contains the codebase, experiments, and configurations for my Master's thesis project on synthetic CT scan data generation conditioned on demographic metadata to evaluate the robustness and fairness of medical AI models.

The project involves downloading and processing real-world datasets, training Latent Diffusion Models, generating synthetic data, training classifiers, evaluating synthetic data, and evaluating model performance across various conditions.

---

## 📁 Project Structure

```
.
├── config/            # Configuration files for all code
│   ├── classifier/        # Configurations for training, tunning, and evaluating a classifier
│   ├── data_filtering/    # Configurations for processing data
│   ├── data_split/        # Configurations for splitting data
│   ├── evaluation/        # Configurations for evaluation framework of the model and the synthetic data 
│   ├── experiments/       # Configurations for the experiments
│   ├── generator/         # Configurations for training, tunning, and sampling a generative model
│   ├── pre_trained/       # Special configurations for fine-tunning pre-trained generative models
├── experiments/       # Experiment for predicting metadata of images
├── notebooks/         # Jupyter notebooks for exploratory analysis, graphing, and prototyping
├── src/               # Main source code
│   ├── checks/            # Scripts to validate data integrity and project setup
│   ├── classifier/        # Classifier model definition, training, and evaluation logic
│   ├── data_download/     # Scripts for downloading dataset from Kaggle
│   ├── data_filtering/    # Filtering logic and pre-processing (e.g., removing non-usable samples, preparing variables, selecting subsets, etc.)
│   ├── data_processing/   # Preprocessing scripts (e.g., DICOM to Nifti, old codebase)
│   ├── data_split/        # Utilities for splitting data into the desired structure
│   ├── datasets/          # Custom dataset for nifti and DICOM (old code)
│   ├── evaluation/        # Evaluation framework for synthetic data and subpopulation analysis
│   │   ├── subpopulation_bias_eval/  # Evaluation scripts for subpopulation bias evaluation of downstream model
│   │   ├── utility/                  # Evaluation scripts for utility of synthetic data
│   │   ├── subpopulation_bias_eval/  # Evaluation scripts for subpopulation bias evaluation of downtream model
│   │   ├── utlity/                   # Evaluation scripts for utlity of synthetic data
│   │   ├── XAI/                      # Scripts to apply XAI methods to results of a downstream model
│   └── generator/         # Synthetic data generation models with different configurations (training from scratch and fine-tunning, sampling)
├── .gitignore          # Git ignore file
├── poetry.lock         # Dependency lock file (Poetry)
├── pyproject.toml      # Project metadata and dependencies
└── README.md           # Project documentation
```

> **Note:**  
> The following folders are used locally but not pushed to GitHub:  
> `.venv/`, `data/`, `logs/`, `models/`, `results/`

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mireiafrt/master-thesis.git
   cd master-thesis
   ```

2. Install dependencies using [Poetry](https://python-poetry.org/):
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

---

## 🚀 Usage

### Configurations
   Edit or create config files in `config/`.

### Prepare Data 
   ```bash
   python src/data_download/kaggle_covid_data_download.py  # need to have Kaggle API Key set up
   python src/data_filtering/data_filter_covid.py
   python src/data_split/data_split_covid.py
   ```

### Generative Model 
   Example:
   ```bash
   python src/generator/aekl_pretrained_finetune.py
   python src/generator/gen_pretrained_finetune.py
   ```

### Evaluate the synthetic data
   Example:
   ```bash
   python src/evaluation/fid_div_metrics/fid_mult_gens_eval.py
   python src/evaluation/utility/general_utility_evaluating.py
   python src/evaluation/utility/general_utility_training.py
   ```

### Evaluate the downstream model
   Example:
   ```bash
   python src/evaluation/subpopulation_bias_eval/subpop_bias_eval.py
   ```

