from __future__ import print_function
import datetime as dt
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import sleep
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Read data from the txt files and plot the times-series data
def read(savedataname1):
    # read data from file
    with open(savedataname1, mode='r') as file:
        reader = csv.reader(file)
        expression_data = list(reader)

    # Convert data to numpy arrays
    expression_data_all = np.array(expression_data)
    # Extract facial expression features and time data
    expression_features = expression_data_all[:, 0:num_features].astype(float)
    time_data = expression_data_all[:, num_features]
    datetime_objs = [dt.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f") for dt_str in time_data]
    floats_time = [(dt_obj - dt.datetime(1970, 1, 1)).total_seconds() for dt_obj in datetime_objs]

    # Subtract the first time from all times
    start = floats_time[0]
    floats_time = [floats_time[i] - start for i in range(len(floats_time))]

    return expression_features, floats_time


# Function for get the first order differentiation of the signal with respect to time
def differentiation(expression_features, floats_time):
    diff = np.gradient(expression_features, axis=0) / np.gradient(floats_time)
    return diff


# Parameters for facial expression recognition
num_features = 68  # Assuming 68 facial landmarks are used as features
num_classes = 9    # Number of facial expression classes

# Segment function for extracting all the useful data pieces.
def segment(expression_features):
    # You can implement the segmentation logic based on your facial expression data.
    # This function should return the beginning indices of the segments.
    # For example:
    # segment_length = int(0.3 * Fs)
    # sorted_sum_amp[:, 1] = sorted_sum_amp[:, 1] - (segment_length * 0.5)
    # return sorted_sum_amp[0:50, 1]
    pass


# Zero the baseline of the signals
def zerobaseline(expression_features, segment_beginning):
    # You can implement the baseline correction based on your facial expression data.
    pass


# Calculate features: mean, standard deviation, minimum, maximum,
# 25th percentile, 50th percentile, 75th percentile, variance, skewness, kurtosis,
# absolute area, sum of squared differences, root mean square
def getfeatures(expression_features, diff, segment_beginning):
    # You can implement feature extraction logic relevant to your facial expression data.
    pass


# This function generates the corresponding labels for the features.
def label(file_name):
    # You can implement the labeling logic based on your facial expression data.
    # For example, if you have labeled facial expression data, create an array of corresponding labels.
    pass


# -------------------------------------------------------------------------------
# Main function starts here.
# The files which need to be analyzed
file_list = ['expression_1.txt', 'expression_2.txt', 'expression_3.txt']

# Calculate all the features for every facial expression
file_number = len(file_list)  # Number of files
row_number = file_number * 50  # Each expression was repeated 50 times
feature_number = 24
column_number = feature_number * 3 * 2  # 3 directions x,y,z for two sets of data mag and diff
features_all = np.zeros((row_number, column_number))
loop_index = 0

# Start calculating all the features.
for filename in file_list:
    expression_features, floats_time = read(filename)
    # Calculate the differentiation
    diff = differentiation(expression_features, floats_time)
    # Segment the signal
    segment_beginning = segment(expression_features)
    # Zero the baseline
    expression_features = zerobaseline(expression_features, segment_beginning)
    # Feature calculation
    features = getfeatures(expression_features, diff, segment_beginning)
    # Store all the features
    features_all[loop_index * 50: loop_index * 50 + 50, :] = features
    loop_index += 1

# Generate an array storing labels
labels = label(file_list)

# LDA training and confusion matrix plot.
# Define number of folds
num_folds = 5

# Create instance of StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize empty confusion matrix
cm = np.zeros((num_classes, num_classes))

# Initialize variable for total correct predictions
total_correct = 0

# Initialize variable for total samples
total_samples = 0

# Loop over folds
for train_idx, test_idx in skf.split(features_all, labels):
    X_train, y_train = features_all[train_idx], labels[train_idx]
    X_test, y_test = features_all[test_idx], labels[test_idx]

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    cm_fold = confusion_matrix(y_test, y_pred)
    cm += cm_fold

    # Update total correct predictions and samples
    total_correct += np.sum(np.diag(cm_fold))
    total_samples += np.sum(cm_fold)

# Calculate the total number of samples in each class
class_totals = np.sum(cm, axis=1)

# Divide each cell of the matrix by the total number of samples in that class
cm_accuracy = cm / class_totals[:, np.newaxis]

# Multiply each cell by 100 to get the percentage accuracy
cm_accuracy *= 100

# Print the accuracy matrix
print(cm_accuracy)

# Print confusion matrix
print(cm)

# Calculate and print accuracy
accuracy = total_correct / total_samples
print('LDA Accuracy:', accuracy)

# Plot confusion matrix
# You can plot the confusion matrix using similar code as shown in the original code.
# You may need to modify the plot labels and titles to match your facial expression classes.

# Micro-averaged ROC curve (if applicable)
# You can calculate and plot the micro-averaged ROC curve similar to the original code if needed.
# The number of classes and labels may vary based on your facial expression data.
