import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# -------------------------------
# 1. Load Data
# -------------------------------
def load_data():
    # Get the directory of this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct full path to the CSV file
    csv_path = os.path.join(BASE_DIR, "data", "customers.csv")

    # Load and return the dataframe
    df = pd.read_csv(csv_path)
    return df


# -------------------------------
# 2. Exploratory Data Analysis
# -------------------------------
def perform_eda(df):
    st.subheader("📊 Exploratory Data Analysis")
    st.write("Data Preview:", df.head())
    st.write("Summary:", df.describe())
    st.bar_chart(df["Region"].value_counts())


# -------------------------------
# 3. Feature Engineering & Modeling
# -------------------------------
def train_model(df):
    # Example: Predicting churn (1 = churn, 0 = active)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Dummy encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    st.subheader("🤖 Model Performance")
    st.write("Accuracy:", acc)
    st.text("Classification Report:\n" + classification_report(y_test, preds))

    return model


# -------------------------------
# 4. Streamlit Dashboard
# -------------------------------
def main():
    st.title("Customer 360 Analytics & Predictive Intelligence Platform")
    st.write("End-to-End Data Science + Analytics + ML Project")

    df = load_data()

    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Modeling", "📈 Insights"])

    with tab1:
        perform_eda(df)

    with tab2:
        model = train_model(df)

    with tab3:
        st.subheader("📈 Business Insights")
        st.write("- Segment customers by region and revenue")
        st.write("- Identify high churn risk customers")
        st.write("- Use predictions for targeted retention campaigns")


if __name__ == "__main__":
    main()
