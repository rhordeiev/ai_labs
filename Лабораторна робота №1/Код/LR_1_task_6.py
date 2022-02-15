import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from utils import visualize_classifier
from sklearn.svm import LinearSVC

# Input file containing data
input_file = 'data_multivar_nb.txt'

# Load data from input file
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Naive Bayes classifier

# Split data into training and test data
X_NB_train, X_NB_test, y_NB_train, y_NB_test = train_test_split(X, y, test_size=0.2, random_state=3)
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_NB_train, y_NB_train)
y_test_NB_pred = naive_bayes_classifier.predict(X_NB_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_NB_test == y_test_NB_pred).sum() / X_NB_test.shape[0]
print("Accuracy of the NB classifier =", round(accuracy, 2), "%")

# Visualize the performance of the classifier
visualize_classifier(naive_bayes_classifier, X_NB_test, y_NB_test)

###############################################
# Scoring functions

num_folds = 3
accuracy_values = cross_val_score(naive_bayes_classifier,
        X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(naive_bayes_classifier,
        X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(naive_bayes_classifier,
        X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(naive_bayes_classifier,
        X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")


print('')


# SVM classifier

# Split data into training and test data
X_SVM_train, X_SVM_test, y_SVM_train, y_SVM_test = train_test_split(X, y, test_size=0.2, random_state=3)
svm_classifier = LinearSVC(dual=False)
svm_classifier.fit(X_SVM_train, y_SVM_train)
y_test_SVM_pred = svm_classifier.predict(X_SVM_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_SVM_test == y_test_SVM_pred).sum() / X_SVM_test.shape[0]
print("Accuracy of the SVM classifier =", round(accuracy, 2), "%")

# Visualize the performance of the classifier
visualize_classifier(svm_classifier, X_SVM_test, y_SVM_test)

###############################################
# Scoring functions

num_folds = 3
accuracy_values = cross_val_score(svm_classifier,
        X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(svm_classifier,
        X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(svm_classifier,
        X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(svm_classifier,
        X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")
