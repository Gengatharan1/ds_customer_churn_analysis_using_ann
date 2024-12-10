import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

# Load Dataset
data = pd.read_csv('Churn_Modelling.csv')

# Step 1: Data Exploration
print("Data Overview:")
print(data.info())
print(data.describe())

# Step 2: Data Cleaning
# Dropping unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Step 3: Encoding Categorical Variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# Step 4: Feature Selection and Scaling
X = data.drop('Exited', axis=1)
y = data['Exited']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Model Training
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = rf_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 8: Insights and Visualization
# Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

# Step 9: Save Processed Data (Optional)
processed_data = pd.DataFrame(X_scaled, columns=feature_names)
processed_data['Exited'] = y.values
processed_data.to_csv('Processed_Churn_Data.csv', index=False)

# Step 10: Streamlit Deployment
st.title("Customer Churn Analysis")

# Load processed data
load_file = st.file_uploader("Upload Processed Data File", type=["csv"])
if load_file is not None:
    processed_data = pd.read_csv(load_file)
    st.write("### Data Overview:")
    st.dataframe(processed_data.head())

    st.write("### Data Description:")
    st.write(processed_data.describe())

    # Visualizations
    st.write("### Churn Distribution:")
    churn_counts = processed_data['Exited'].value_counts()
    st.bar_chart(churn_counts)

    st.write("### Feature Importance:")
    feature_importance_chart = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    st.write(feature_importance_chart)

    # Recommendations
    st.write("### Recommendations:")
    st.markdown("- Implement targeted loyalty programs for older customers.")
    st.markdown("- Encourage inactive members to engage through personalized campaigns.")
    st.markdown("- Focus retention efforts in regions with higher churn rates, especially Germany.")
