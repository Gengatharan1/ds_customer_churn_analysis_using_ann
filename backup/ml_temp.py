# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------
# 1. Load and Preprocess Data
# ----------------------------------------------
@st.cache
def load_data(file_path):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)
    return data

# Upload data file
st.title("Customer Churn Prediction Dashboard")
uploaded_file = st.file_uploader("Upload Processed Data File", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("### Sample Data:")
    st.dataframe(data.head())

    # Split the data
    X = data.drop(columns=['Exited', 'CustomerId', 'Surname', 'RowNumber'], axis=1)
    y = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------------
    # 2. Train Predictive Models
    # ----------------------------------------------
    # Logistic Regression Model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    # Random Forest Model
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # ----------------------------------------------
    # 3. Evaluate Models
    # ----------------------------------------------
    st.write("### Model Evaluation")
    
    # Logistic Regression Metrics
    st.write("#### Logistic Regression Metrics:")
    st.text(classification_report(y_test, y_pred_log))
    st.write(f"ROC-AUC Score: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])}")

    # Random Forest Metrics
    st.write("#### Random Forest Metrics:")
    st.text(classification_report(y_test, y_pred_rf))
    st.write(f"ROC-AUC Score: {roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])}")

    # Feature Importance for Random Forest
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.write("#### Random Forest Feature Importance:")
    st.dataframe(feature_importance)

    # ----------------------------------------------
    # 4. Predict and Visualize
    # ----------------------------------------------
    st.write("### Predictions")
    predictions = rf_model.predict(X)
    data['Predicted_Churn'] = predictions

    # Show churn distribution
    churn_counts = data['Predicted_Churn'].value_counts()
    st.write("#### Churn Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax)
    ax.set_title("Predicted Churn Distribution")
    ax.set_xlabel("Churn (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # ----------------------------------------------
    # 5. Targeted Interventions
    # ----------------------------------------------
    st.write("### Recommendations for Targeted Interventions")
    st.write("""
    - **Loyalty Programs**: Engage older customers with loyalty rewards and exclusive offers.
    - **Boost Activity**: Design campaigns to re-engage inactive customers.
    - **Regional Focus**: Prioritize high-risk regions like Germany for customer retention efforts.
    """)

    # ----------------------------------------------
    # 6. Save Processed Data with Predictions
    # ----------------------------------------------
    save_file = st.checkbox("Save Processed Data with Predictions")
    if save_file:
        data.to_csv("churn_predictions.csv", index=False)
        st.success("File saved as `churn_predictions.csv`")

# ----------------------------------------------
# Additional Notes for Deployment
# ----------------------------------------------
st.write("### Real-Time Monitoring")
st.write("""
Integrate this dashboard with live data feeds to monitor customer churn in real time and update predictions dynamically.
""")
