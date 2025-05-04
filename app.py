import streamlit as st
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Small clean font style
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 14px;
    }
    h1 { font-size: 28px !important; }
    h2 { font-size: 22px !important; }
    h3 { font-size: 18px !important; }
    .dataframe tbody tr th, .dataframe thead th {
        font-size: 13px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💳 Credit Card Fraud Detection")

# Select algorithm
model_option = st.selectbox("Select a Machine Learning Model", [
    "Logistic Regression",
    "Decision Tree",
    "Support Vector Machine (SVM)",
    "Naive Bayes",
    "K-Nearest Neighbors (KNN)"
])

# Upload file
uploaded_file = st.file_uploader("Upload your credit card transaction CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(data.head(5))

    if 'Class' not in data.columns:
        st.error("❌ The CSV must contain a 'Class' column for fraud labels (0 = legit, 1 = fraud).")
    else:
        # Balance the dataset
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]
        legit_sample = legit.sample(n=len(fraud), random_state=1)
        balanced_data = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=1)

        X = balanced_data.drop(columns='Class', axis=1)
        Y = balanced_data['Class']

        # Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Choose model
        if model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
        elif model_option == "Support Vector Machine (SVM)":
            model = SVC()
        elif model_option == "Naive Bayes":
            model = GaussianNB()
        elif model_option == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=5)

        # Train model
        model.fit(X_train, Y_train)

        # Predict
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(Y_train, train_pred)
        test_acc = accuracy_score(Y_test, test_pred)

        st.subheader("📈 Model Performance")
        st.write(f"🔹 *Training Accuracy:* {train_acc:.4f}")
        st.write(f"🔹 *Testing Accuracy:* {test_acc:.4f}")
        st.success(f"🕵‍♀ Fraud Detected in Test Data: {sum(test_pred)} transactions")

        # 📋 Styled Classification Report
        # 📋 Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(Y_test, test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        report_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
        st.dataframe(report_df)

        # 📉 Clean & Compact Confusion Matrix
        # 📉 Confusion Matrix — only for models except Decision Tree
        if model_option != "Decision Tree":
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(Y_test, test_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(3, 2.5))  # Small and neat
            sns.heatmap(cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=False,
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"],
                        annot_kws={"size": 10})
            ax_cm.set_xlabel("Predicted", fontsize=10)
            ax_cm.set_ylabel("Actual", fontsize=10)
            ax_cm.set_title("Confusion Matrix", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_cm)

        # 🌳 Show Decision Tree (if selected)
        if model_option == "Decision Tree":
            st.subheader("🌳 Visualized Decision Tree")
            fig, ax = plt.subplots(figsize=(16, 8))
            plot_tree(model,
                      feature_names=X.columns,
                      class_names=["Legit", "Fraud"],
                      filled=True, rounded=True, fontsize=10, ax=ax)
            st.pyplot(fig)