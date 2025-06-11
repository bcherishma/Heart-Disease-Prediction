import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# App title
st.title("Heart Disease Prediction ")


# Upload dataset
uploaded_file = st.file_uploader("Upload your heart disease dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader(" Data Preview")
    st.dataframe(data.head())
        # Patients with Heart Disease
    st.subheader(" Patients with Heart Disease")
    st.dataframe(data[data['target'] == 1])


    # Numeric distribution
    st.subheader("Feature Distributions")
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    cols = 3
    rows = (len(numerical_features) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        sns.histplot(data=data, x=feature, hue='target', kde=True, ax=axes[i], multiple='stack')
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # Remove empty plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)

    # Age boxplot
    st.subheader("Age Distribution by Heart Disease")
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(x='target', y='age', data=data)
    plt.title('Age vs. Heart Disease')
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(fig)

    # Preprocessing
    st.subheader(" Model Training & Evaluation")

    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    model_choice = st.selectbox("Select a model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train and predict
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write("Classification Report")
    st.text(classification_report(y_test, predictions))

    st.write(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    # Confusion matrix
    st.write(" Confusion Matrix")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions,
        cmap="Blues" if model_choice == "Logistic Regression" else "Greens")
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Feature importances
    if model_choice == "Random Forest":
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        features = X.columns
        sorted_indices = np.argsort(importances)[::-1]

        fig = plt.figure(figsize=(8, 6))
        plt.barh(range(len(importances)), importances[sorted_indices], align='center', color='teal')
        plt.yticks(range(len(importances)), [features[i] for i in sorted_indices])
        plt.xlabel("Importance")
        plt.title("Random Forest Feature Importances")
        plt.gca().invert_yaxis()
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file containing the heart dataset.")
