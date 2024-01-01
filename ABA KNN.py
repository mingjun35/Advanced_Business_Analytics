#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:58:53 2023

@author: leemingjun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:46:46 2023

@author: leemingjun
"""


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
from io import StringIO


def KNN_model(dataset, target_col):
    
    np.random.seed(42)
        
    # Create a new DataFrame without the target column
    X = dataset.drop(target_col, axis=1)

    # Extract the target variable
    y = dataset[target_col]

    # Split the data into training and testing sets at 70:30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create an instance of the DecisionTreeClassifier
    knn = KNeighborsClassifier()

    # Fit the decision tree model
    knn.fit(X_train, y_train)
    
    # Make predictions using the trained model
    predictions = knn.predict(X_test)
    
    # Convert predictions into a pandas DataFrame and set the column name to 'Churn'
    #predictions_rename = pd.DataFrame(predictions, columns=['Churn'])
    
    predictions_rename = pd.Series(predictions, name = 'Churn')
    
    return predictions_rename, y_test

target_col = 'Churn'

KNN_predictions, KNN_y_test = KNN_model(data_aba_final, target_col)




######## Confusion Matrix ########

Confusion_Matrix_DF_KNN = pd.DataFrame(columns=["True Positive", "False Positive", 
                                                "False Negative", "True Negative"])

KNN_matrix = confusion_matrix(KNN_y_test, KNN_predictions)

###### For Showing Only! #######
KNN_confusion_matrix = pd.DataFrame(KNN_matrix, index=['No', 'Yes'], columns=['No', 'Yes'])

print(KNN_confusion_matrix)

# Extract true positive, false positive, false negative, and true negative values from the confusion matrix
true_positive = KNN_matrix[1, 1]
false_positive = KNN_matrix[0, 1]
false_negative = KNN_matrix[1, 0]
true_negative = KNN_matrix[0, 0]


# Create a temporary DataFrame with the current iteration results
temp_df = pd.DataFrame({"True Positive": [true_positive],
                        "False Positive": [false_positive],
                        "False Negative": [false_negative],
                        "True Negative": [true_negative]})

# Concatenate the temporary DataFrame with the Confusion_Matrix_DF_DT DataFrame
Confusion_Matrix_DF_KNN = pd.concat([Confusion_Matrix_DF_KNN, temp_df], ignore_index=True)

print(Confusion_Matrix_DF_KNN)



######## Calculate f1-score, precision, sensitivity, specificity ########

Performance_DF_KNN = pd.DataFrame(columns=["Accuracy", "F1-Score"])


def calculate_metrics(confusion_matrix_df):
    
    # Calculate accuracy
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    # Calculate precision
    precision = true_positive / (true_positive + false_positive)
        
    # Calculate sensitivity (recall)
    sensitivity = true_positive / (true_positive + false_negative)
        
    # Calculate specificity
    specificity = true_negative / (true_negative + false_positive)
        
    # Calculate F1-score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        
    return accuracy, f1_score


for index, row in Confusion_Matrix_DF_KNN.iterrows():
    true_positive = row["True Positive"]
    false_positive = row["False Positive"]
    false_negative = row["False Negative"]
    true_negative = row["True Negative"]

accuracy, f1_score = calculate_metrics(Confusion_Matrix_DF_KNN)

# Create a temporary DataFrame with the current iteration results
temporary_df = pd.DataFrame({"Accuracy": [accuracy],
                             "F1-Score": [f1_score]})

# Concatenate the temporary DataFrame with the Performance_DF_DT DataFrame
Performance_DF_KNN = pd.concat([Performance_DF_KNN, temporary_df], ignore_index=True)

print(Performance_DF_KNN)