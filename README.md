# 📞 MR - Telecom Customer Churn Prediction

A **Streamlit web application** for predicting customer churn in the telecom sector. This app uses a pre-trained machine learning pipeline to estimate the probability of a customer leaving (churning) and provides actionable retention suggestions.

---

## 🧩 Features

- **Login/Signup System:** Secure authentication using JSON-based user storage.
- **Customer Input Form:** Enter or auto-fill random customer details for prediction.
- **Churn Prediction:** Predicts whether a customer is likely to churn (`CHURN 🔴`) or stay (`STAY 🟢`).
- **Probability Visualization:**
  - Gauge chart for churn probability
  - Bar chart comparing Stay vs Churn
- **Feature Analysis:** Interactive plots for categorical features vs churn.
- **Correlation Heatmap:** Visualize relationships between numeric features.
- **Retention Suggestions:** Personalized tips to reduce churn risk.
- **Download Options:**
  - CSV history of predictions
  - PDF report for individual predictions (requires ReportLab)

---

## ⚡ Technologies Used

- **Python 3.x**
- **Streamlit** – Web app framework
- **Pandas** – Data manipulation
- **Joblib** – Loading ML pipeline
- **Plotly, Matplotlib, Seaborn** – Data visualization
- **ReportLab** – PDF report generation (optional)
- **SHAP** – Model explainability (optional)

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mr-telecom-churn.git
cd mr-telecom-churn

##Install dependencies:
pip install -r requirements.txt

##Ensure you have the following files in the project folder:

main.py – Streamlit app
churn_pipeline.pkl – Pre-trained ML model pipeline
WA_Fn-UseC_-Telco-Customer-Churn.csv – Dataset
users.json – Optional, for login credentials

##Run the Streamlit app:

streamlit run main.py
Login or signup as a new user.
Fill in the customer details manually or auto-fill a random customer.
Click Predict Churn Now! to get prediction results.
Download prediction history as CSV or a PDF report for individual predictions.


##How It Works

The app loads a pre-trained churn prediction pipeline and the dataset.
Users provide customer details via the input form.
The model predicts churn probability:
CHURN 🔴 if probability ≥ 50%
STAY 🟢 if probability < 50%
Results are visualized with interactive charts.
Personalized retention suggestions are generated based on customer attributes.

📊 Visualization Features

Gauge Chart: Displays churn probability dynamically.
Bar Chart: Compares Stay vs Churn probability.
Categorical Feature Plots: Shows feature distribution against churn.
Correlation Heatmap: Displays relationships between numeric features.

⚠️ Requirements

Python ≥ 3.8
Streamlit
Pandas, Numpy
Plotly, Matplotlib, Seaborn
Joblib
Optional:
ReportLab (for PDF report generation)
SHAP (for explainable AI features)

##🏷 License

This project is licensed under the MIT License.
