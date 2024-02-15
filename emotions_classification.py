import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
import numpy as np

# Load the dataset
file_path = 'Emotion_classify_Data.csv'
data = pd.read_csv(file_path)

# Calculate the total number of samples
total_samples = data.shape[0]

# Calculate the number of samples per class
samples_per_class = data['Emotion'].value_counts()

print(f"Total number of samples: {total_samples}")
print("Number of samples per emotion class:")
print(samples_per_class)


# Add transformation of emotions to numeric values using Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Emotion_encoded'] = label_encoder.fit_transform(data['Emotion'])

# Text preprocessing
tfidf_vectorizer = TfidfVectorizer()

# Initial dataset split between training+validation and testing (test set)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    data['Comment'],
    data['Emotion_encoded'],
    test_size=0.2,   # Allocate 20% for testing
    random_state=42
)

# Train the Naive Bayes model
model = MultinomialNB()
X_train_transformed = tfidf_vectorizer.fit_transform(X_train_val)
X_test_transformed = tfidf_vectorizer.transform(X_test)
model.fit(X_train_transformed, y_train_val)

# Predictions and evaluation for Naive Bayes
nb_y_pred = model.predict(X_test_transformed)  # Use nb_y_pred instead of y_pred
nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_report = classification_report(y_test, nb_y_pred)

# Display results
print("Accuracy (test set):", nb_accuracy)
print("Classification Report (test set):\n", nb_report)

# Apply KNN algorithm
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as preferred
knn_model.fit(X_train_transformed, y_train_val)

# Evaluate KNN algorithm
knn_y_pred = knn_model.predict(X_test_transformed)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_report = classification_report(y_test, knn_y_pred)

# Display results for KNN
print("KNN Accuracy (test set):", knn_accuracy)
print("KNN Classification Report (test set):\n", knn_report)

def predict_sentiment(model, vectorizer, text, model_type='NB'):
    text_transformed = vectorizer.transform([text])
    if model_type == 'NB':
        predicted_sentiment = model.predict(text_transformed)
    elif model_type == 'KNN':
        predicted_sentiment = model.predict(text_transformed)
    else:
        raise ValueError("Invalid model type. Choose 'NB' for Naive Bayes or 'KNN' for K-Nearest Neighbors.")
    return predicted_sentiment[0]

# Predict sentiment based on an entered comment
comment = "Being stuck in traffic for hours really tests my patience."
nb_predicted_sentiment = predict_sentiment(model, tfidf_vectorizer, comment, model_type='NB')
print("Sentiment predicted for NB is (test set): ", nb_predicted_sentiment)
knn_predicted_sentiment = predict_sentiment(knn_model, tfidf_vectorizer, comment, model_type='KNN')
print("Sentiment predicted for KNN is (test set): ", knn_predicted_sentiment)

# Functions for saving results and confusion matrix
def save_confusion_matrix(y_true, y_pred, class_labels, file_name="confusion_matrix.png"):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (test set)')
    plt.savefig(os.path.join('./results', file_name))
    plt.close()
    print(f'Confusion matrix saved at ./results/{file_name}')

def save_results(y_true, y_pred, file_name="results.csv"):
    results_df = pd.DataFrame({
        'True Label': y_true,
        'Predicted Label': y_pred
    })
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_file = os.path.join(results_folder, file_name)
    results_df.to_csv(results_file, index=False)
    print(f'Results saved at {results_file}')

# Define class_labels using LabelEncoder
class_labels = label_encoder.classes_

# Transform y_test into strings
y_test_str = label_encoder.inverse_transform(y_test)

# Transform y_pred into strings if necessary
if isinstance(nb_y_pred[0], int):
    nb_y_pred_str = label_encoder.inverse_transform(nb_y_pred)
else:
    nb_y_pred_str = nb_y_pred

if isinstance(knn_y_pred[0], int):
    knn_y_pred_str = label_encoder.inverse_transform(knn_y_pred)
else:
    knn_y_pred_str = knn_y_pred

# Both sets are in string format for confusion matrix
y_test_str = label_encoder.inverse_transform(y_test)
nb_y_pred_str = label_encoder.inverse_transform(nb_y_pred)
knn_y_pred_str = label_encoder.inverse_transform(knn_y_pred)

# Calculate confusion matrix
nb_conf_matrix = confusion_matrix(y_test_str, nb_y_pred_str, labels=label_encoder.classes_)
knn_conf_matrix = confusion_matrix(y_test_str, knn_y_pred_str, labels=label_encoder.classes_)

# Display confusion matrix
sns.heatmap(nb_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Naive Bayes Confusion Matrix (test set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('KNN Confusion Matrix (test set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Call functions to save Naive Bayes results
save_results(y_test_str, nb_y_pred_str, "naive_bayes_results_test_set.csv")
save_confusion_matrix(y_test_str, nb_y_pred_str, class_labels, "naive_bayes_confusion_matrix_test_set.png")

# Call functions to save KNN results
save_results(y_test_str, knn_y_pred_str, "knn_results_validation set.csv")
save_confusion_matrix(y_test_str, knn_y_pred_str, class_labels, "knn_confusion_matrix_test_set.png")

# Split the training+validation set into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,  # From the remaining 80%, 25% will go to validation and 75% to training
    random_state=42
)

# Train the Naive Bayes model
model = MultinomialNB()
X_train_transformed = tfidf_vectorizer.fit_transform(X_train)
X_test_transformed = tfidf_vectorizer.transform(X_test)
model.fit(X_train_transformed, y_train)

# Predictions and evaluation for Naive Bayes
nb_y_pred = model.predict(X_test_transformed)  # Use nb_y_pred instead of y_pred
nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_report = classification_report(y_test, nb_y_pred)

# Display results
print("Accuracy (validation set):", nb_accuracy)
print("Classification Report (validation set):\n", nb_report)

# Apply KNN algorithm
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as preferred
knn_model.fit(X_train_transformed, y_train)

# Evaluate KNN algorithm
knn_y_pred = knn_model.predict(X_test_transformed)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_report = classification_report(y_test, knn_y_pred)

# Display results for KNN
print("KNN Accuracy (validation set):", knn_accuracy)
print("KNN Classification Report (validation set):\n", knn_report)

def predict_sentiment(model, vectorizer, text, model_type='NB'):
    text_transformed = vectorizer.transform([text])
    if model_type == 'NB':
        predicted_sentiment = model.predict(text_transformed)
    elif model_type == 'KNN':
        predicted_sentiment = model.predict(text_transformed)
    else:
        raise ValueError("Invalid model type. Choose 'NB' for Naive Bayes or 'KNN' for K-Nearest Neighbors.")
    return predicted_sentiment[0]

# Predict sentiment based on an entered comment
comment = "Being stuck in traffic for hours really tests my patience."
nb_predicted_sentiment = predict_sentiment(model, tfidf_vectorizer, comment, model_type='NB')
print("Sentiment predicted for NB is (validation set): ", nb_predicted_sentiment)
knn_predicted_sentiment = predict_sentiment(knn_model, tfidf_vectorizer, comment, model_type='KNN')
print("Sentiment predicted for KNN is (validation set): ", knn_predicted_sentiment)

# Functions for saving results and confusion matrix
def save_confusion_matrix(y_true, y_pred, class_labels, file_name="confusion_matrix_training_set.png"):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (validation set)')
    plt.savefig(os.path.join('./results', file_name))
    plt.close()
    print(f'Confusion matrix saved at ./results/{file_name}')

def save_results(y_true, y_pred, file_name="results_validation_set.csv"):
    results_df = pd.DataFrame({
        'True Label': y_true,
        'Predicted Label': y_pred
    })
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_file = os.path.join(results_folder, file_name)
    results_df.to_csv(results_file, index=False)
    print(f'Results saved at {results_file}')

# Define class_labels using LabelEncoder
class_labels = label_encoder.classes_

# Transform y_test into strings
y_test_str = label_encoder.inverse_transform(y_test)

# Transform y_pred into strings if necessary
if isinstance(nb_y_pred[0], int):
    nb_y_pred_str = label_encoder.inverse_transform(nb_y_pred)
else:
    nb_y_pred_str = nb_y_pred

if isinstance(knn_y_pred[0], int):
    knn_y_pred_str = label_encoder.inverse_transform(knn_y_pred)
else:
    knn_y_pred_str = knn_y_pred

# Both sets are in string format for confusion matrix
y_test_str = label_encoder.inverse_transform(y_test)
nb_y_pred_str = label_encoder.inverse_transform(nb_y_pred)
knn_y_pred_str = label_encoder.inverse_transform(knn_y_pred)

# Calculate confusion matrix
nb_conf_matrix = confusion_matrix(y_test_str, nb_y_pred_str, labels=label_encoder.classes_)
knn_conf_matrix = confusion_matrix(y_test_str, knn_y_pred_str, labels=label_encoder.classes_)

# Display confusion matrix
sns.heatmap(nb_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Naive Bayes Confusion Matrix (validation set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('KNN Confusion Matrix (validation set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Call functions to save Naive Bayes results
save_results(y_test_str, nb_y_pred_str, "naive_bayes_results_validation_set.csv")
save_confusion_matrix(y_test_str, nb_y_pred_str, class_labels, "naive_bayes_confusion_matrix_validation_set.png")

# Call functions to save KNN results
save_results(y_test_str, knn_y_pred_str, "knn_results_validation_test.csv")
save_confusion_matrix(y_test_str, knn_y_pred_str, class_labels, "knn_confusion_matrix_validation_set.png")
