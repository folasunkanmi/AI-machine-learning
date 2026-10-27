# AI Machine Learning - Customer Churn Prediction

A complete end-to-end **Customer Churn Prediction** project using Python and machine learning.

This repository was previously empty and has been fully set up with a ready-to-run project for predicting customer churn based on typical telecom/customer features.

## Project Overview

Customer churn is when a customer stops doing business with a company. Predicting churn helps businesses take proactive retention actions.

**What this project includes:**
- Synthetic but realistic customer dataset generation
- Exploratory Data Analysis (EDA)
- Data preprocessing & feature engineering
- Multiple machine learning models (Logistic Regression, Random Forest, Gradient Boosting)
- Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance analysis
- Model saving for future use

## Tech Stack

- Python 3.9+
- Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Joblib (for model persistence)

## Project Structure

```
AI-machine-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── churn_prediction.py          # Main end-to-end script
└── models/                      # Saved models (created after running)
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/folasunkanmi/AI-machine-learning.git
   cd AI-machine-learning
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis**
   ```bash
   python churn_prediction.py
   ```

The script will:
- Generate a synthetic customer dataset
- Perform EDA and print key insights
- Train and evaluate multiple models
- Save the best model to `models/best_churn_model.joblib`
- Display feature importance

## Dataset Features (Synthetic)

| Feature              | Description                              |
|----------------------|------------------------------------------|
| tenure               | Months the customer has stayed           |
| monthly_charges      | Monthly bill amount                      |
| total_charges        | Total amount charged                     |
| contract_type        | Month-to-month / One year / Two year     |
| payment_method       | Electronic / Mailed / Bank transfer / Credit card |
| internet_service     | DSL / Fiber / No                         |
| online_security      | Yes / No                                 |
| tech_support         | Yes / No                                 |
| senior_citizen       | 0 or 1                                   |
| partner              | Yes / No                                 |
| dependents           | Yes / No                                 |
| **churn**            | Target: Yes / No                         |

## Models Used

1. **Logistic Regression** – Interpretable baseline
2. **Random Forest** – Strong non-linear performance
3. **Gradient Boosting** – High accuracy ensemble

The best model (by F1-score / ROC-AUC) is automatically selected and saved.

## Next Steps / Ideas to Improve

- Replace synthetic data with real Telco Customer Churn dataset from Kaggle
- Add hyperparameter tuning (GridSearchCV / Optuna)
- Build a simple Streamlit or FastAPI demo
- Add SHAP for model explainability
- Deploy the model

## Author

**Folasunkanmi**  
GitHub: [folasunkanmi](https://github.com/folasunkanmi)

---

Feel free to fork, star, or open issues if you want to extend this project!
