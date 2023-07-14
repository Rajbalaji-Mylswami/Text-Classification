# Toxic Tweets Classification using NLP

This project aims to classify toxic and non-toxic tweets using Natural Language Processing (NLP) techniques. The dataset used for this project is the [Toxic Tweets Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset) available on Kaggle.


## Dependencies

To run this project, you need the following dependencies:

- streamlit
- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn

## Usage

1. Run the Streamlit app:

2. Access the app in your browser at `http://localhost:8501`.

3. Select the method of coverting text using the dropdown.

4. Choose a machine learning model from the dropdown.

5. Explore the classification report, confusion matrix, and ROC curve with AUC for the selected model.


## Dataset

The dataset used in this project contains a collection of tweets labeled as toxic (1) or non-toxic (0). It is provided in the `FinalBalancedDataset.csv` file. The dataset has been preprocessed by applying text cleaning techniques, such as removing special characters, converting text to lowercase, removing stop words, and tokenizing the text.

## Text Conversion Methods

Two text conversion methods are available:

1. **Bag of Words**: This method represents text data as a matrix of word counts, where each row represents a document (tweet) and each column represents a unique word in the corpus.
2. **TF-IDF**: This method represents text data as a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) values, which take into account the importance of words in the document and the entire corpus.

## Classification Models

The following classification models are available for predicting the toxicity of tweets:

- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors
- Support Vector Machine (SVM)

## Evaluation Metrics

For each selected classification model, the application provides the following evaluation metrics:

- Precision: the ratio of true positive predictions to the total predicted positives.
- Recall: the ratio of true positive predictions to the total actual positives.
- F1-score: the weighted average of precision and recall.
- Confusion Matrix: a matrix showing the number of true positive, false positive, true negative, and false negative predictions.
- ROC-AUC Curve: a plot showing the receiver operating characteristic (ROC) curve and the area under the curve (AUC) score.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request.
