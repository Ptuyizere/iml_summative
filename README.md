# Financial Inclusion in Africa — Bank Account Ownership Prediction

A machine-learning project predicting bank account ownership from demographic and socioeconomic survey data across East Africa (Kenya, Rwanda, Tanzania, and Uganda). This work benchmarks traditional machine-learning models (Logistic Regression, Random Forest, Extra Trees) against Feedforward Neural Networks, with dedicated evaluation on class imbalance handling and country-level performance.

---

## Project Overview

- **Problem Type:** Binary Classification (predicting bank account ownership: `Yes` / `No`)
- **Key Challenges:** Severe class imbalance, demographic variance across multiple countries, tabular feature engineering
- **Core Models:** Logistic Regression, Random Forest, Extra Trees, Feedforward Neural Networks (MLP)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision (PR-AUC)

---

## Dataset & Links

- **Dataset:** [Financial Inclusion in Africa — Zindi](https://zindi.africa/competitions/financial-inclusion-in-africa) *(Kaggle mirror also available)*.  

---

## Repository Contents

```text
├── financial_inclusion_africa.ipynb   # Complete project notebook
├── requirements.txt                   # Environment Python package dependencies
└── README.md                          # Project documentation and reproduction guide
```

---

## Setup & Installation

Clone the repository and install the required dependencies in a virtual environment:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Dependencies
- **Python:** `3.9+`
- **Core Libraries:** `numpy`, `pandas`, `scipy`
- **Machine Learning & Deep Learning:** `scikit-learn`, `tensorflow`, `joblib`
- **Data Visualization:** `matplotlib`, `seaborn`

---

## Reproduction Steps

1. **Download Data:** Obtain `Train.csv` directly from the [Zindi competition page](https://zindi.africa/competitions/financial-inclusion-in-africa) or the Kaggle mirror.
2. **Place File:** Place `Train.csv` in the root directory of this project, or update the `DATA_PATH` constant in **Section 3** of the notebook.
3. **Execute Pipeline:** Open `financial_inclusion_africa.ipynb` and run all cells sequentially.
4. **Export Artifacts:** To persist models and tables, execute **Section 26**. When executed in Google Colab, this automatically syncs serialized models and experiment outputs to your mounted Google Drive directory.

---

## Generated Outputs

Upon pipeline completion, the following artifacts are generated:

| Artifact | Format | Description |
| :--- | :--- | :--- |
| `best_traditional_model.joblib` | Joblib file | Best tuned tree-based model bundled with the fitted preprocessing pipeline (`ColumnTransformer`). |
| `best_nn_model.keras` | Keras format | Selected neural network architecture and trained weights. |
| `experiment_table.csv` | CSV | Validation benchmark metrics across all tested models and hyperparameter variations. |

---
