# Customer Churn Prediction

**End-to-end Machine Learning Project** | Python • Scikit-learn • Data Analysis

Predict which customers are likely to leave a telecom/service company so the business can take proactive retention actions.

> **Important Note on Data**: This project uses a **synthetic (artificially generated)** dataset designed to realistically mimic customer behaviour. It is excellent for demonstrating the full machine learning workflow, but the results should not be treated as findings from real customer data.

---

## Business Problem

Customer churn is expensive. Acquiring a new customer typically costs 5–7× more than retaining an existing one.  
By identifying customers at high risk of leaving, a company can:

- Offer targeted discounts or loyalty offers
- Improve service for at-risk segments
- Reduce revenue loss
- Increase customer lifetime value

This project builds a predictive model that flags high-risk customers so the business can act before they churn.

---

## Project Highlights

- Full machine learning pipeline (data generation → EDA → preprocessing → modelling → evaluation)
- Three models compared: **Logistic Regression**, **Random Forest**, and **Gradient Boosting**
- Clear evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Feature importance analysis to understand *why* customers churn
- Business-oriented interpretation of results
- Ready-to-run code with saved model

---

## Tech Stack

- Python 3.9+
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Joblib

---

## How to Run

```bash
git clone https://github.com/folasunkanmi/AI-machine-learning.git
cd AI-machine-learning
pip install -r requirements.txt
python churn_prediction.py
```

The script will:
1. Generate a realistic synthetic customer dataset
2. Perform Exploratory Data Analysis
3. Train and evaluate three models
4. Select and save the best model
5. Show feature importance and key insights

---

## Models Compared

| Model                  | Strengths                              | Typical Use Case                  |
|------------------------|----------------------------------------|-----------------------------------|
| Logistic Regression    | Highly interpretable, fast             | Baseline & explainability         |
| Random Forest          | Strong performance, handles non-linearity | Robust general performance     |
| Gradient Boosting      | Highest predictive power               | Best overall accuracy             |

The best model is automatically selected based on **F1-Score** (balanced for imbalanced churn data) and saved for future use.

---

## Key Business Insights (from the model)

Typical high-risk signals the model learns:
- Month-to-month contracts
- High monthly charges + short tenure
- No online security or tech support
- Fiber optic internet (higher expectations / competition)
- Senior citizens or customers without partners/dependents

**How a business can use this:**
1. Score all current customers monthly
2. Flag the top 10–20% highest-risk customers
3. Trigger retention campaigns (special offers, personal outreach, service upgrades)
4. Measure reduction in churn rate over time

---

## Project Structure

```
AI-machine-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── churn_prediction.py      # Complete end-to-end pipeline
└── models/                  # Saved best model (created after running)
```

---

## Future Improvements

- Replace synthetic data with the real Telco Customer Churn dataset (Kaggle)
- Add hyperparameter tuning (GridSearchCV / Optuna)
- Create an interactive Streamlit dashboard for business users
- Add SHAP values for individual customer explanations
- Deploy as an API

---

## Author

**Folasunkanmi**  
Data Analyst & Aspiring Data Scientist  
Skills: Python • SQL • Excel • Power BI • Machine Learning  
GitHub: [folasunkanmi](https://github.com/folasunkanmi)

---

⭐ If you find this project useful, feel free to star the repository!
