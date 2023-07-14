import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# Disable the warning about the use of st.pyplot() without arguments
st.set_option('deprecation.showPyplotGlobalUse', False)


def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def preprocess_text(df):
    df['tweet'] = df['tweet'].apply(text_cleaning)
    return df


def convert_to_bag_of_words(data):
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(data)
    return bag_of_words


def convert_to_tfidf(data):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(data)
    return tfidf


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name):
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Naive Bayes":
        model = MultinomialNB()
    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    elif model_name == "SVM":
        model = SVC(kernel='linear', probability=True)
    else:
        return None

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Compute performance metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(y_test)

    # Compute accuracy scores
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # Display accuracy scores
    st.write("Training Accuracy Score:", train_accuracy)
    st.write("Testing Accuracy Score:", test_accuracy)

    # Display performance metrics
    st.subheader("Classification Report:")
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.write(report)

    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm)
    st.subheader("Confusion Matrix:")
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    st.pyplot()

    # Compute ROC-AUC if the model supports probability prediction
    if hasattr(model, "predict_proba"):
        # Compute the predicted probabilities for the positive class
        y_scores = model.predict_proba(X_test)[:, 1]

        # Compute the ROC curve and ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # Display ROC curve
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.subheader("ROC Curve:")
        st.pyplot(fig)


def main():
    # Set title and description
    st.title("Text Classification and Evaluation")

    # Load the data from CSV and preprocess the text
    df = pd.read_csv(r"C:\Users\shrib\Downloads\archive (5)\FinalBalancedDataset.csv")
    df = preprocess_text(df)

    # Drop the 'Unnamed: 0' column
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Create a dropdown for selecting the method to convert the text
    text_conversion_method = st.selectbox("Convert Text using", ["Bag of Words", "TF-IDF"])

    # Convert the tweet column using the selected method
    if text_conversion_method == "Bag of Words":
        X = convert_to_bag_of_words(df["tweet"])
    else:
        X = convert_to_tfidf(df["tweet"])

    # Set the target variable as 'toxicity'
    y = df['Toxicity']

    # Create a dropdown for selecting the model
    model_name = st.selectbox("Choose a model",
                              ["Decision Tree", "Random Forest", "Naive Bayes", "K-Nearest Neighbors", "SVM"])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the selected model
    train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name)


if __name__ == "__main__":
    main()
