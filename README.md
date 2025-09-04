# Credit Fraud Detection

## Overview
This project implements a credit card fraud detection model using machine learning techniques on an imbalanced dataset. The dataset contains anonymized credit card transactions, with features transformed via PCA (V1 to V28), along with Time, Amount, and a Class label (0 for legitimate, 1 for fraudulent). The goal is to detect fraudulent transactions accurately, focusing on metrics like Precision-Recall AUC due to the severe class imbalance (only ~0.17% fraud cases).

Key steps include:
- Data preprocessing (scaling, handling imbalance with SMOTE).
- Hyperparameter tuning using Optuna for an LSTM neural network.
- Evaluation on a test set, achieving high accuracy and good fraud recall.

This notebook (`credit_risk.ipynb`) demonstrates the end-to-end process.

## Dataset
- Source: `creditcard.csv` (not included in repo; download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) or similar).
- Features: 30 (Time, V1-V28 from PCA, Amount).
- Target: Class (binary: 0 = legitimate, 1 = fraud).
- Size: 284,807 transactions.
- Imbalance: 284,315 legitimate vs. 492 fraudulent.

## Requirements
- Python 3.x
- Libraries: 
  - numpy
  - pandas
  - imblearn (for SMOTE)
  - matplotlib, seaborn (for visualization)
  - sklearn (for preprocessing, splitting, metrics)
  - tensorflow/keras (for LSTM model)
  - optuna (for hyperparameter tuning)

Install via:
```
pip install numpy pandas imbalanced-learn matplotlib seaborn scikit-learn tensorflow optuna
```

## Usage
1. Download and place `creditcard.csv` in the project directory.
2. Open `credit_risk.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the cells sequentially:
   - Load and explore data.
   - Preprocess (scale, split, oversample with SMOTE).
   - Tune hyperparameters with Optuna (runs 20 trials; adjust `n_trials` as needed).
   - Train the final LSTM model with best params.
   - Evaluate on test set.
4. Results will include PR AUC, classification report, and accuracy.

Example output from evaluation:
```
Test PR AUC: 0.8989
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.88      0.87      0.87        98

    accuracy                           1.00     56962
   macro avg       0.94      0.93      0.94     56962
weighted avg       1.00      1.00      1.00     56962

Test Accuracy: 0.9996
```

## Model Details
- **Architecture**: LSTM neural network with tunable units (e.g., 60), dropout (e.g., ~0.41), and learning rate (e.g., ~0.00023).
- **Optimizer**: Adam.
- **Loss**: Binary Crossentropy.
- **Metrics**: Accuracy, Precision-Recall AUC (optimized via Optuna).
- **Training**: 18 epochs, batch size 16, validation split 0.2.
- **Handling Imbalance**: SMOTE oversampling on training data.

## Results
- High overall accuracy (~99.96%) due to imbalance.
- Strong fraud detection: ~87% recall and precision on minority class.
- PR AUC: ~0.90, indicating good performance on imbalanced data.

## Limitations
- Dataset is anonymized, so feature interpretation is limited.
- Model assumes similar distribution in new data; retrain if needed.
- Optuna tuning can be computationally intensive; reduce trials for faster runs.
- Error in notebook (KeyboardInterrupt) suggests training was interrupted—ensure sufficient resources.

## Contributing
Fork the repo, make changes, and submit a pull request. For issues, open a GitHub issue.

## Credits
- Dataset: ULB Machine Learning Group.
- Inspired by https://ieeexplore.ieee.org/document/10541645
- Tools: Built with Keras, Optuna, and scikit-learn.

---
