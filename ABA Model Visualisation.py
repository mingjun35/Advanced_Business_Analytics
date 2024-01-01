#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:25:25 2023

@author: leemingjun
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Add a new column to each dataframe to indicate the type of predictive modeling
Performance_DF_DT['Model'] = 'Decision Tree'
Performance_DF_LR['Model'] = 'Logistic Regression'
Performance_DF_RF['Model'] = 'Random Forest'

# Concatenate the dataframes vertically
Merged_DF = pd.concat([Performance_DF_DT, Performance_DF_LR, Performance_DF_RF], ignore_index=True)



######## Comparison Graph ########

# Set the width of the bars
bar_width = 0.30

# Set the positions of the bars on the x-axis
bar_positions = np.arange(len(Merged_DF))

# Plotting the bars
fig, ax = plt.subplots(figsize=(8, 6))
accuracy_bars = ax.bar(bar_positions - bar_width/2, Merged_DF['Accuracy'], bar_width, label='Accuracy')
f1_bars = ax.bar(bar_positions + bar_width/2, Merged_DF['F1-Score'], bar_width, label='F1-score')

# Add labels, title, and axes ticks
ax.set_xlabel('Modelling Technique')
ax.set_ylabel('Performance (%)')
ax.set_title('Accuracy and F1-score Comparison')



ax.set_xticks(bar_positions)
ax.set_xticklabels(Merged_DF['Model'])
ax.legend(bbox_to_anchor=(1.02, 1))

# Add value labels to the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

add_value_labels(accuracy_bars)
add_value_labels(f1_bars)

# Display the chart
plt.tight_layout()
plt.show()



######## Confusion Matrix ########

#Decision Tree

sns.heatmap(DT_confusion_matrix, annot=True, cmap='Greens', fmt='g', linecolor="k", linewidths=1.5)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()


#Logistic Regression

sns.heatmap(LR_confusion_matrix, annot=True, cmap='Reds', fmt='g', linecolor="k", linewidths=1.5)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


#Random Forest

sns.heatmap(RF_confusion_matrix, annot=True, cmap='Blues', fmt='g', linecolor="k", linewidths=1.5)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()


######## ROC Chart ########


# Calculate predicted probabilities for Decision Tree and Logistic Regression
y_DT_pred_prob = Tree.predict_proba(DT_X_test)[:, 1]
y_LR_pred_prob = LogisticRegression.predict_proba(LR_X_test)[:, 1]
y_RF_pred_prob = RandomForest.predict_proba(RF_X_test)[:,1]

# Calculate ROC curves and AUC scores for all models
fpr_dt, tpr_dt, thresholds_dt = roc_curve(DT_y_test, y_DT_pred_prob)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(LR_y_test, y_LR_pred_prob)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(RF_y_test, y_RF_pred_prob)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.plot(fpr_rf, tpr_rf, label='Random Forest', color='b')
plt.plot(fpr_dt, tpr_dt, label='Decision Tree', color='g')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression', color='r')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Random Forest, Decision Tree, and Logistic Regression', fontsize=16)
plt.legend()
plt.show()



######## Observing Individual Predictors on Churn ########

pastel_colors = ['#FFB6C1', '#FFD700', '#87CEEB', '#98FB98', '#FFA07A', '#FF69B4', '#00BFFF', 
                 '#FFE4E1', '#FF6347', '#40E0D0', '#F0E68C', '#DDA0DD', '#8FBC8F', '#87CEFA']

pastel_colors_1 = ['#DDA0DD', '#FFB6C1', '#FF69B4', '#98FB98', '#FF6347', '#F5DEB3', '#87CEEB', 
                   '#FFE4E1', '#D8BFD8', '#00BFFF', '#87CEFA', '#FFD700', '#8FBC8F', '#DEB887', 
                   '#E6E6FA', '#F0E68C', '#40E0D0', '#FFA07A', '#ADD8E6', '#FFC0CB']

pastel_colors_2 = ['#F49AC2', '#F6999D', '#F3C27A', '#FFDB58', '#7ECC85', '#74C365', '#B3B3D7', 
                   '#9C89B8', '#C9C9FF', '#9FE2BF', '#96DED1', '#CBF3F0', '#D4A5A5', '#FAB2E1', 
                   '#FFC3A0', '#F1CB00', '#BBFFBB', '#FFCCCC', '#C0C0C0', '#E0E0E0']

######## Decision Tree ########

DT_importances = Tree.feature_importances_
DT_weights = pd.Series(DT_importances,
                 index=DT_X.columns.values)
DT_top_features = DT_weights.sort_values(ascending = False)[:10]

DT_top_features.plot(kind='bar', color=pastel_colors)

# Adding labels and title
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances from Decision Tree Model')


######## Logistic Regression ########


LR_coefficient = LogisticRegression.coef_[0]
LR_weights = pd.Series(LR_coefficient, index=LR_X.columns.values)
LR_top_features = LR_weights.sort_values(ascending = False)[:10]

LR_top_features.plot(kind='bar', color=pastel_colors_1)

# Adding labels and title
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances from Logistic Regression Model')


######## Random Forest ########

RF_importances = RandomForest.feature_importances_
RF_weights = pd.Series(RF_importances,
                 index=RF_X.columns.values)
RF_top_features = RF_weights.sort_values(ascending = False)[:10]

RF_top_features.plot(kind='bar', color=pastel_colors_2)

# Adding labels and title
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances from Random Forest Model')
