import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import json
import os
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Optional deps
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORT_AVAILABLE = True
except Exception:
    REPORT_AVAILABLE = False

# =============================
# 1️⃣ Load Trained Pipeline & Dataset
# =============================
st.set_page_config(page_title="📞 Fancy Churn Predictor", layout="wide")

# Load model & dataset
pipeline = joblib.load("churn_pipeline.pkl")
raw_df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

if 'customerID' in raw_df.columns:
    raw_df = raw_df.drop(columns=['customerID'])

# TotalCharges cleanup
if "TotalCharges" in raw_df.columns:
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    mask_na = raw_df["TotalCharges"].isna()
    raw_df.loc[mask_na, "TotalCharges"] = (raw_df.loc[mask_na, "MonthlyCharges"].fillna(0) *
                                           raw_df.loc[mask_na, "tenure"].fillna(0))
    raw_df["TotalCharges"] = raw_df["TotalCharges"].fillna(0)

raw_df['Churn'] = raw_df['Churn'].map({'No': 0, 'Yes': 1})

# Features
target = "Churn"
X = raw_df.drop(columns=[target])
y = raw_df[target]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

if "TotalCharges" not in numeric_features:
    numeric_features.append("TotalCharges")
if "TotalCharges" in categorical_features:
    categorical_features.remove("TotalCharges")

if "SeniorCitizen" in numeric_features:
    numeric_features.remove("SeniorCitizen")
if "SeniorCitizen" not in categorical_features:
    categorical_features.append("SeniorCitizen")

features = X.columns.tolist()

# =============================
# 2️⃣ Auth Helpers
# =============================
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {"admin": "admin123"}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

users = load_users()
ss = st.session_state
ss.setdefault("authenticated", False)
ss.setdefault("username", "")
ss.setdefault("history", [])
ss.setdefault("form_values", {})

# =============================
# 3️⃣ Styling
# =============================
st.markdown("""
<style>
.stApp {background-color: #1b1b1b; color: white;}
h1,h2,h3,h4,h5,h6 {color: #0d3b66;}
p, span, label {color: white;}
.stButton>button {
    background: linear-gradient(90deg, #ff69b4, #ff1493);
    color: white; font-weight: 700; font-size: 18px;
    border-radius: 12px; padding: 12px 25px; margin-top: 10px;
    border: 0; cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s, filter 0.2s;
}
.stButton>button:hover {transform: translateY(-1px); box-shadow: 0 0 15px #ff1493; filter: brightness(1.05);}
.accent>button { background: linear-gradient(90deg, #16a085, #1abc9c) !important; }
.danger>button { background: linear-gradient(90deg, #ff5252, #ff1744) !important; }
.muted>button  { background: linear-gradient(90deg, #607d8b, #455a64) !important; }
.card {
    padding: 25px; border-radius: 20px;
    background: linear-gradient(135deg, #0d3b66, #1b1b1b);
    box-shadow: 4px 4px 20px #444; margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {transform: scale(1.02); box-shadow: 6px 6px 25px #666;}
.section {margin-bottom: 20px;}
.streamlit-expanderHeader {
    font-size: 18px; font-weight: bold; color: #ff69b4 !important;
    background: linear-gradient(135deg, #0d3b66, #1b1b1b); padding: 10px; border-radius: 12px;
}
.streamlit-expanderContent {
    padding: 15px; background: linear-gradient(135deg, #0d3b66, #1b1b1b);
    border-radius: 12px; box-shadow: 3px 3px 15px #444; transition: all 0.3s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 4️⃣ Helpers
# =============================
def retention_tips(inp: dict, prob: float):
    tips = []
    if inp.get("Contract", "") == "Month-to-month":
        tips.append("Lock-in offer: 3–6 month discount to move them off month-to-month.")
    if inp.get("PaymentMethod", "") == "Electronic check":
        tips.append("Encourage auto-pay (bank/credit) with a small incentive to reduce churn risk.")
    if inp.get("TechSupport", "") == "No":
        tips.append("Offer free premium tech support for 1–2 months.")
    if float(inp.get("MonthlyCharges", 0)) > 80:
        tips.append("Review plan to a lower-cost bundle; highlight same-value alternatives.")
    if inp.get("InternetService", "") == "Fiber optic":
        tips.append("Bundle streaming add-ons or loyalty rewards for high-value fiber customers.")
    if not tips:
        tips.append("Maintain proactive engagement: periodic check-ins, usage tips, loyalty offers.")
    if prob >= 0.5:
        tips.insert(0, "Risk is high — contact via call/SMS within 24h and personalize retention offer.")
    return tips[:5]

def get_random_customer():
    row = X.sample(1).iloc[0].to_dict()
    if "SeniorCitizen" in row:
        row["SeniorCitizen"] = "Yes" if int(row["SeniorCitizen"]) == 1 else "No"
    for k in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if k in row:
            try: row[k] = float(row[k])
            except: row[k] = 0.0
    return row

def build_pdf_report(buffer, username, input_dict, prob, pred_label):
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Telecom Customer Churn Prediction", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated by: {username or 'user'}", styles['Normal']))
    story.append(Paragraph(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Prediction: <b>{pred_label}</b>", styles['Heading2']))
    story.append(Paragraph(f"Churn Probability: <b>{prob:.2f}%</b>", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Inputs:", styles['Heading3']))
    for k, v in input_dict.items():
        story.append(Paragraph(f"{k}: {v}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Retention Tips:", styles['Heading3']))
    for tip in retention_tips(input_dict, prob/100):
        story.append(Paragraph(f"• {tip}", styles['Normal']))
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    doc.build(story)

# =============================
# 5️⃣ Login / Signup
# =============================
if not ss["authenticated"]:
    st.title("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])
    with tab1:
        username = st.text_input("👤 Username", key="login_user")
        password = st.text_input("🔑 Password", type="password", key="login_pass")
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Login"):
                if username in users and users[username] == password:
                    ss["authenticated"] = True
                    ss["username"] = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    with tab2:
        new_user = st.text_input("👤 New Username", key="signup_user")
        new_pass = st.text_input("🔑 New Password", type="password", key="signup_pass")
        if st.button("Signup"):
            if not new_user or not new_pass:
                st.error("Please enter a username and password")
            elif new_user in users:
                st.error("⚠️ Username already exists")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("✅ Account created! Please login.")
# =============================
# 6️⃣ Main App (After Login)
# =============================
else:
    st.title("📞 Telecom Customer Churn Prediction")
    st.markdown(f"👋 Welcome, **{ss['username']}**")

    top_col1, top_col2, top_col3 = st.columns([1,1,2])
    with top_col1:
        if st.container().button("🚪 Logout", key="logout_btn"):
            ss["authenticated"] = False
            ss["username"] = ""
            st.success("✅ Logged out successfully!")
            st.rerun()
    with top_col2:
        if st.container().button("🎲 Auto-Fill Random Customer", key="autofill_btn"):
            ss["form_values"] = get_random_customer()
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------- Input Form ----------
    with st.expander("📋 Enter Customer Details", expanded=True):
        user_input = {}
        defaults = ss.get("form_values", {})

        for col in numeric_features:
            if col == "tenure":
                user_input[col] = st.number_input(col, min_value=0, max_value=1_000_000,
                                                  value=int(defaults.get(col, 0)), step=1, key=f"in_{col}")
            elif col in ["MonthlyCharges", "TotalCharges"]:
                user_input[col] = st.number_input(col, min_value=0.0, max_value=1_000_000.0,
                                                  value=float(defaults.get(col, 0.0)), step=1.0, key=f"in_{col}")
            else:
                col_min, col_max = 0.0, 1_000_000.0
                def_val = float(defaults.get(col, col_min))
                user_input[col] = st.number_input(col, min_value=col_min, max_value=col_max,
                                                  value=def_val, key=f"in_{col}")

        for col in categorical_features:
            options = raw_df[col].dropna().astype(str).unique().tolist()
            if col == "SeniorCitizen":
                senior_choice = defaults.get("SeniorCitizen", "No")
                choice = st.selectbox("SeniorCitizen (Yes = 1, No = 0)", ["No", "Yes"],
                                      index=1 if senior_choice == "Yes" else 0, key="in_SeniorCitizen")
                user_input["SeniorCitizen"] = 1 if choice == "Yes" else 0
            else:
                def_opt = str(defaults.get(col, options[0] if options else ""))
                if def_opt not in options and options:
                    def_opt = options[0]
                user_input[col] = st.selectbox(col, options, index=options.index(def_opt) if options else 0,
                                               key=f"in_{col}")

        ss["form_values"] = {**user_input, "SeniorCitizen": "Yes" if user_input.get("SeniorCitizen", 0)==1 else "No"}

    input_df = pd.DataFrame([user_input], columns=features)

    # ---------- Prediction & Visualization ----------
    c_pred1, c_pred2, c_pred3 = st.columns([1.2, 1, 1])
    with c_pred1:
        do_pred = st.button("🚀 Predict Churn Now!", key="predict_btn")
    with c_pred2:
        download_csv = st.button("⬇️ Download History (CSV)", key="dl_hist")
    with c_pred3:
        pdf_btn = st.button("🧾 Download PDF Report", key="dl_pdf")

    if do_pred:
        probability = float(pipeline.predict_proba(input_df)[:, 1][0])
        probability_percent = round(probability * 100, 2)
        prediction_text = "CHURN 🔴" if probability >= 0.5 else "STAY 🟢"
        color = "red" if probability >= 0.5 else "green"

        # Save history
        hist_row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": ss['username'],
                    "prediction": "CHURN" if probability >= 0.5 else "STAY",
                    "probability(%)": probability_percent}
        for k, v in user_input.items():
            hist_row[k] = v
        ss["history"].append(hist_row)

        # Prediction Card
        st.markdown(f"""
            <div class="card section">
                <h2>Prediction: {prediction_text}</h2>
                <p>Churn Probability: {probability_percent}% | Stay Probability: {round(100 - probability_percent, 2)}%</p>
            </div>
        """, unsafe_allow_html=True)

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability_percent,
            number={'suffix': '%'},
            delta={'reference': 50, 'increasing': {'color': 'red'}, 'decreasing': {'color': 'green'}},
            title={'text': "Churn Probability"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': color},
                   'steps': [{'range': [0, 50], 'color': 'lightgreen'},
                             {'range': [50, 100], 'color': 'lightcoral'}]}))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Bar Chart
        fig_bar = go.Figure(go.Bar(
            x=["Stay", "Churn"],
            y=[100 - probability_percent, probability_percent],
            marker_color=["green", "red"],
            text=[f"{100 - probability_percent}%", f"{probability_percent}%"],
            textposition='auto'
        ))
        fig_bar.update_layout(title="Stay vs Churn Probability", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Categorical Churn Plots
        st.subheader("📊 Feature vs Churn Analysis")
        for col in categorical_features:
            fig_cat = px.histogram(raw_df, x=col, color="Churn", barmode='group', height=300)
            st.plotly_chart(fig_cat, use_container_width=True)

        # Correlation Heatmap
        st.subheader("📈 Correlation Heatmap")
        corr = raw_df.select_dtypes(include=['float64', 'int64']).corr()
        fig_corr, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)

     

        # Retention Tips
        tips = retention_tips(user_input, probability)
        st.info("💡 **Retention Suggestions:**\n- " + "\n- ".join(tips))

    # ---------- CSV Download ----------
    if download_csv:
        if ss["history"]:
            hist_df = pd.DataFrame(ss["history"])
            csv_bytes = hist_df.to_csv(index=False).encode()
            st.download_button("Download CSV", data=csv_bytes, file_name="churn_prediction_history.csv", mime="text/csv")
            st.dataframe(hist_df)
        else:
            st.info("No predictions yet.")

    # ---------- PDF Download ----------
    if pdf_btn:
        if not ss["history"]:
            st.info("Run a prediction first to generate a PDF.")
        else:
            last = ss["history"][-1]
            prob_pct = float(last["probability(%)"])
            pred_lbl = "CHURN" if last["prediction"] == "CHURN" else "STAY"
            if REPORT_AVAILABLE:
                buffer = io.BytesIO()
                pdf_input = {k: v for k, v in last.items() if k not in ["timestamp", "user", "prediction", "probability(%)"]}
                scv = pdf_input.get("SeniorCitizen", 0)
                pdf_input["SeniorCitizen"] = "Yes" if int(scv) == 1 else "No"
                build_pdf_report(buffer, ss["username"], pdf_input, prob_pct, pred_lbl)
                st.download_button("Download PDF", data=buffer.getvalue(),
                                   file_name="churn_report.pdf", mime="application/pdf")
            else:
                st.warning("PDF export requires ReportLab. Install: `pip install reportlab`")
