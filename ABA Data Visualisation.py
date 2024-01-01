#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:22:32 2023

@author: leemingjun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


######## Data Visualisation ########


######## Observe Correlation of Predictors with Churn ########


plt.figure(figsize=(15,8))

# Calculate the correlation of predictors with Churn
correlations = data_aba_final.corr()['Churn'].sort_values(ascending=False)

# Define a color palette with different colors for each bar
pastel_colors = ['#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', 
                 '#6a3d9a', '#ffff99', '#b15928', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']


# Plot the bar chart with the specified color palette
correlations.plot(kind='bar', color=pastel_colors)

plt.xlabel('Predictors')

plt.xlabel('Predictors')
plt.ylabel('Correlation with Churn')
plt.title('Correlation of Predictors with Churn')
plt.show()




######## Monthly Charges by Churn ########


######## #KDE Plot #########

sns.set_context("paper",font_scale=1.1)

ax = sns.kdeplot(data_aba.MonthlyCharges[(data_aba["Churn"] == 'No') ], color="Green", shade = True)

ax = sns.kdeplot(data_aba.MonthlyCharges[(data_aba["Churn"] == 'Yes') ], ax =ax, color="Red", shade= True)

ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Kernel Density Estimation');
ax.set_xlabel('Monthly Charges');
ax.set_title('Distribution of Monthly Charges by Churn');


######## Histogram with Frequency ########

sns.set_context("paper", font_scale=1.1)

ax = sns.histplot(data_aba.MonthlyCharges[(data_aba["Churn"] == 'No')], color="Green", alpha=0.5, bins=30)
ax = sns.histplot(data_aba.MonthlyCharges[(data_aba["Churn"] == 'Yes')], ax=ax, color="Red", alpha=0.5, bins=30)

ax.legend(["Not Churn", "Churn"], loc='upper right')
ax.set_ylabel('Frequency')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of Monthly Charges by Churn')
plt.show()




######## Total Charges by Churn ########

######## #KDE Plot #########

ax = sns.kdeplot(data_aba.TotalCharges[(data_aba["Churn"] == 'No') ], color="Blue", shade = True);

ax = sns.kdeplot(data_aba.TotalCharges[(data_aba["Churn"] == 'Yes') ], ax =ax, color="Red", shade= True);

ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Total Charges');
ax.set_title('Distribution of Total Charges by Churn');


######## Histogram with Frequency ########

ax = sns.histplot(data_aba.TotalCharges[(data_aba["Churn"] == 'No') ], color="Blue", alpha=0.5, bins=30)

ax = sns.histplot(data_aba.TotalCharges[(data_aba["Churn"] == 'Yes') ], ax =ax, color="Red", alpha=0.5, bins=30)

ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Frequency');
ax.set_xlabel('Total Charges');
ax.set_title('Distribution of Total Charges by Churn');
