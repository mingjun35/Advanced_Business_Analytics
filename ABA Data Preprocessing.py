#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:10:17 2023

@author: leemingjun
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler




data_aba=pd.read_csv('/Users/leemingjun/Desktop/Telco-Customer-Churn.csv')

# Drop the "customer id" column
data_aba.drop(columns=['customerID'], inplace=True)



######## Data Preprocessing ########


######## Check for Null Values ########

#Convert Total Charges Column to numeric, non-numeric are converted to NaN Values
data_aba['TotalCharges'] = pd.to_numeric(data_aba.TotalCharges, errors='coerce')
data_aba.isnull().sum()


######## Identify Rows with NaN Values ########

# Filter rows where 'TotalCharges' column contains NaN
nan_data = data_aba[data_aba['TotalCharges'].isnull()]

# Print the rows with NaN values
print(nan_data)


######## Impute NaN Values ########

# List of columns with missing values
missing_col = data_aba.columns[data_aba.isnull().any()].tolist()


# Apply mode imputation to the missing columns
imputer = SimpleImputer(strategy='mean')

data_aba[missing_col] = imputer.fit_transform(data_aba[missing_col])


######## Check again from missing values ########

data_aba.isnull().sum()


######## Create a new dataset for encoded categorical columns ########


data_aba_dummy = data_aba.copy()

data_aba_dummy['Churn'].replace(to_replace='Yes', value=1, inplace=True)
data_aba_dummy['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Let's convert all the categorical variables into dummy variables
data_aba_dummy = pd.get_dummies(data_aba_dummy)



# Scaling all the variables to a range of 0 to 1

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the scaled features
data_aba_scaled = scaler.fit_transform(data_aba_dummy)


data_aba_final = pd.DataFrame(data_aba_scaled, columns=data_aba_dummy.columns)

data_aba_final.dtypes


#data_aba.to_csv('/Users/leemingjun/Desktop/Telco Customer Churn.csv')


#data_aba_final.to_csv('/Users/leemingjun/Desktop/Telco Customer Churn Encoded.csv')

