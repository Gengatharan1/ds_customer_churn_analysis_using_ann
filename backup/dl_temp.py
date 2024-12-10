# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------
# 1. Load and Preprocess Data
# ----------------------------------------------
@st.cache
def load_data(file_path):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)
    return data

# Upload file
st.title("Customer Churn Prediction Using ANN")
uploaded_file = st.file_uploader("Upload Processed Data File", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("### Sample Data:")
    st.dataframe(data.head())

    # Split dataset
    X = data.drop(columns=['Exited', 'CustomerId', 'Surname', 'RowNumber'], axis=1)
    y = data['Exited']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ----------------------------------------------
    # 2. Build and Train ANN
    # ----------------------------------------------
    def create_ann_model(input_dim):
        """Create an ANN model."""
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.3))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=1, activation='sigmoid'))  # Binary classification output
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Build model
    ann_model = create_ann_model(X_train.shape[1])
    st.write("### Training ANN Model...")
    history = ann_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

    # ----------------------------------------------
    # 3. Evaluate ANN Model
    # ----------------------------------------------
    st.write("### Model Evaluation")
    y_pred_prob = ann_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Classification metrics
    st.write("#### Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    st.write("#### Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC-AUC Score
    roc_score = roc_auc_score(y_test, y_pred_prob)
    st.write(f"#### ROC-AUC Score: {roc_score:.2f}")

    # ----------------------------------------------
    # 4. Visualize Training Performance
    # ----------------------------------------------
    st.write("### Training Performance")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Plot
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss Over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Accuracy Plot
    ax[1].plot(history.history['accuracy'], label='Train Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title("Accuracy Over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    st.pyplot(fig)

    # ----------------------------------------------
    # 5. Predict and Save Results
    # ----------------------------------------------
    st.write("### Predictions on Entire Dataset")
    predictions = (ann_model.predict(X_scaled) > 0.5).astype(int)
    data['Predicted_Churn'] = predictions

    # Show churn distribution
    churn_counts = data['Predicted_Churn'].value_counts()
    st.write("#### Predicted Churn Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax)
    ax.set_title("Predicted Churn Distribution")
    ax.set_xlabel("Churn (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Save processed data with predictions
    save_file = st.checkbox("Save Processed Data with Predictions")
    if save_file:
        data.to_csv("churn_predictions_ann.csv", index=False)
        st.success("File saved as `churn_predictions_ann.csv`")

# ----------------------------------------------
# 6. Recommendations for Churn Reduction
# ----------------------------------------------
st.write("### Recommendations")
st.write("""
- **Engage Older Customers**: Develop loyalty programs for older customers to reduce churn.
- **Re-Engage Inactive Members**: Implement targeted campaigns for inactive users to increase engagement.
- **Focus on High-Churn Regions**: Concentrate efforts on regions like Germany for churn reduction.
""")
