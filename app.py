
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "churn_model.pkl")
    threshold = joblib.load(BASE_DIR / "churn_threshold.pkl")
except FileNotFoundError:
    st.error(
        "Model files not found. Place churn_model.pkl and "
        "churn_threshold.pkl in the same folder as app.py."
    )
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0;
    }
    .subtitle {
        color: #808495;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 16px;
        border-radius: 12px;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Churn Intelligence")
st.sidebar.caption("Customer Retention Analytics")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Predict Churn", "Model Performance", "Batch Prediction", "About Project"]
)

st.sidebar.divider()
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write(f"**Decision threshold:** {threshold:.2f}")
st.sidebar.caption("Predictions are estimates, not guarantees.")

# ---------------- HELPER FUNCTION ----------------
def get_risk(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def prepare_customer(
    gender, senior, partner, dependents, tenure,
    phone, multiple_lines, internet, security, backup,
    protection, tech_support, streaming_tv, streaming_movies,
    contract, paperless, payment, monthly, total
):
    return pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])


def predict_customer(customer):
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(probability >= threshold)
    return probability, prediction


def display_prediction(probability, prediction):
    risk = get_risk(probability)

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{probability:.1%}")
    col2.metric("Risk Category", risk)
    col3.metric(
        "Model Decision",
        "Likely to Churn" if prediction else "Likely to Stay"
    )

    st.progress(min(max(probability, 0.0), 1.0))

    if prediction:
        st.warning(
            "This customer is flagged as likely to churn at the "
            f"current threshold of {threshold:.0%}. Consider reviewing "
            "their contract, support experience, and pricing."
        )
    else:
        st.success(
            "This customer is below the current churn decision threshold."
        )

    if risk == "High Risk":
        st.info("Suggested action: prioritize a retention review.")
    elif risk == "Medium Risk":
        st.info("Suggested action: review engagement and service satisfaction.")
    else:
        st.info("Suggested action: maintain service quality and engagement.")


# ---------------- DASHBOARD PAGE ----------------
if page == "Dashboard":
    # st.markdown(
    #     '<p class="main-title">Customer Churn Dashboard</p>',
    #     unsafe_allow_html=True
    # 
    st.header("📊 Customer Churn Dashboard")
    st.markdown(
        '<p class="subtitle">Understand customer retention and explore churn patterns.</p>',
        unsafe_allow_html=True
    )

    data_path = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not data_path.exists():
        st.info(
            "Add WA_Fn-UseC_-Telco-Customer-Churn.csv to this folder "
            "to enable dataset analytics. You can still use the prediction pages."
        )
    else:
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        if "Churn" not in df.columns:
            st.error("The dataset must contain a 'Churn' column.")
            st.stop()

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(),
            errors="coerce"
        ).fillna(0)

        churn_count = (df["Churn"].astype(str).str.strip() == "Yes").sum()
        total_customers = len(df)
        churn_rate = (
            churn_count / total_customers * 100
            if total_customers else 0
        )
        avg_monthly = df["MonthlyCharges"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total_customers:,}")
        c2.metric("Churned Customers", f"{churn_count:,}")
        c3.metric("Churn Rate", f"{churn_rate:.1f}%")
        c4.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")

        st.divider()
        left, right = st.columns(2)

        with left:
            st.subheader("Customer Churn Distribution")
            churn_counts = (
                df["Churn"].astype(str).str.strip()
                .value_counts()
                .reindex(["No", "Yes"], fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x=churn_counts.index,
                y=churn_counts.values,
                ax=ax
            )
            ax.set_xlabel("Churn")
            ax.set_ylabel("Number of Customers")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.0f", padding=3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Churn by Contract Type")
            if "Contract" in df.columns:
                contract_data = (
                    df.groupby("Contract", observed=False)["Churn"]
                    .apply(
                        lambda x: (
                            x.astype(str).str.strip() == "Yes"
                        ).mean() * 100
                    )
                    .sort_values(ascending=False)
                )

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    x=contract_data.index,
                    y=contract_data.values,
                    ax=ax
                )
                ax.set_xlabel("Contract Type")
                ax.set_ylabel("Churn Rate (%)")
                ax.tick_params(axis="x", rotation=15)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.1f%%", padding=3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Monthly Charges vs. Churn")
        if "MonthlyCharges" in df.columns:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.boxplot(
                data=df,
                x="Churn",
                y="MonthlyCharges",
                order=["No", "Yes"],
                ax=ax
            )
            ax.set_xlabel("Churn")
            ax.set_ylabel("Monthly Charges ($)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("Preview dataset"):
            st.dataframe(df.head(20), use_container_width=True)

# ---------------- SINGLE CUSTOMER PREDICTION ----------------
elif page == "Predict Churn":
    st.title("🔍 Predict Customer Churn")
    st.write("Enter customer details to estimate their churn probability.")

    with st.form("customer_form"):
        st.subheader("Customer Profile")
        a, b, c = st.columns(3)

        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with b:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )

        with c:
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges ($)", min_value=0.0,
                max_value=200.0, value=70.0, step=1.0
            )
            total = st.number_input(
                "Total Charges ($)", min_value=0.0,
                max_value=10000.0, value=840.0, step=10.0
            )

        submitted = st.form_submit_button(
            "Predict Churn", type="primary", use_container_width=True
        )

    if submitted:
        customer = prepare_customer(
            gender, senior, partner, dependents, tenure,
            phone, multiple_lines, internet, security, backup,
            protection, tech_support, streaming_tv, streaming_movies,
            contract, paperless, payment, monthly, total
        )

        try:
            probability, prediction = predict_customer(customer)
            display_prediction(probability, prediction)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption(
                "Check that the saved model was trained using the same "
                "19 input features and compatible scikit-learn version."
            )

# ---------------- BATCH PREDICTION ----------------
elif page == "Batch Prediction":
    st.title("📁 Batch Customer Prediction")
    st.write(
        "Upload a CSV with the same 19 input columns used to train your model. "
        "Do not include the Churn target column."
    )

    st.caption(
        "Required columns must match the training dataset names exactly. "
        "Values must use the same categories as the original dataset."
    )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df.columns = batch_df.columns.str.strip()

            required_columns = list(model.feature_names_in_)
            missing_columns = [
                col for col in required_columns
                if col not in batch_df.columns
            ]

            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}")
            else:
                batch_input = batch_df[required_columns].copy()

                if batch_input.isnull().any().any():
                    st.error(
                        "Your CSV contains missing values. Please clean "
                        "them before uploading."
                    )
                else:
                    probabilities = model.predict_proba(batch_input)[:, 1]
                    batch_df["Churn Probability"] = probabilities
                    batch_df["Churn Prediction"] = [
                        "Yes" if p >= threshold else "No"
                        for p in probabilities
                    ]
                    batch_df["Risk Category"] = [
                        get_risk(p) for p in probabilities
                    ]

                    st.success(f"Processed {len(batch_df)} customers.")
                    st.dataframe(batch_df, use_container_width=True)

                    csv_data = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_data,
                        file_name="customer_churn_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Could not process the uploaded file: {e}")

# ---------------- MODEL EVALUATION PAGE ----------------
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write(
        "Evaluate how well the Logistic Regression model "
        "identifies customers who may churn."
    )

    data_path = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not data_path.exists():
        st.error("Dataset CSV not found in the project folder.")
        st.stop()

    df_eval = pd.read_csv(data_path)
    df_eval.columns = df_eval.columns.str.strip()

    # Clean the data
    df_eval["TotalCharges"] = pd.to_numeric(
        df_eval["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    ).fillna(0)

    # Prepare features and target
    X = df_eval.drop(columns=["customerID", "Churn"], errors="ignore")
    y = (
        df_eval["Churn"]
        .astype(str)
        .str.strip()
        .map({"No": 0, "Yes": 1})
    )

    # Match the original training split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Ensure feature order matches the saved model
    X_test = X_test[list(model.feature_names_in_)]

    # Use the existing trained model; do not retrain it here
    y_probability = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probability >= threshold).astype(int)

    # KPI metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    auc = roc_auc_score(y_test, y_probability)

    st.caption(
        f"Evaluation threshold: {threshold:.2f} | "
        f"Test customers: {len(y_test):,}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{accuracy:.1%}")
    c2.metric("Precision", f"{precision:.1%}")
    c3.metric("Recall", f"{recall:.1%}")
    c4.metric("F1-Score", f"{f1:.1%}")
    c5.metric("ROC-AUC", f"{auc:.3f}")

    st.divider()

    # Confusion matrix and ROC curve
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Stay", "Predicted Churn"],
            yticklabels=["Actual Stay", "Actual Churn"],
            ax=ax
        )
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "False negatives are customers who actually churned "
            "but were predicted to stay."
        )

    with right:
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, y_probability)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Stay (0)", "Churn (1)"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style.format("{:.3f}"),
        use_container_width=True
    )

    st.info(
        "**Business insight:** Recall measures the proportion of "
        "actual churners detected by the model. Precision measures "
        "how many customers flagged for churn actually churned. "
        "The right balance depends on the cost of retention campaigns."
    )


# ---------------- ABOUT PAGE ----------------
elif page == "About Project":
    st.title("ℹ️ About Customer Churn Intelligence")

    st.markdown("""
    ### Project Overview
    This machine learning application estimates whether a telecom customer
    is likely to leave the service.

    ### Technology Stack
    - **Python:** application logic
    - **Pandas & NumPy:** data processing
    - **Scikit-learn:** preprocessing and Logistic Regression
    - **Matplotlib & Seaborn:** visual analytics
    - **Streamlit:** interactive web application
    - **Joblib:** saved model loading

    ### Model Notes
    - The application uses the saved model and decision threshold.
    - Churn probability is an estimate, not certainty.
    - Retention suggestions are general recommendations, not causal findings.
    """)
