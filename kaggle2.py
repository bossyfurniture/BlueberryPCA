# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

from altair import *


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


cols = list(pd.read_csv('C:/Users/qwerty/Downloads/archive/WildBlueberryPollinationSimulationData.csv', nrows=1))


# Read the CSV file
df = pd.read_csv('C:/Users/qwerty/Downloads/archive/WildBlueberryPollinationSimulationData.csv', usecols =[i for i in cols if i != "Row#"])


# Perform operations on the DataFrame
# For example, print the first few rows

# Read column names from file

print(df.head())
df_train, df_test= train_test_split(df, test_size=0.33, random_state=42)

# Assuming you have your train and test data stored in X_train and X_test respectively

# Standardize the data
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)

# Create a PCA object and specify the number of components to retain
n_components = 2  # Specify the number of components you want to retain
pca = PCA(n_components=n_components)

# Fit the PCA model on the training data
pca.fit(df_train_scaled)

# Transform the training and test data using the fitted PCA model
df_train_pca = pca.transform(df_train_scaled)
df_test_pca = pca.transform(df_test_scaled)

# Print the explained variance ratio (optional) 
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

#plt.scatter(df_train_pca[:,0],df_train_pca[:,1])
#plt.colorbar()
#plt.show

trainyield = df_train['yield'].to_numpy()
trainyield = trainyield.reshape(520,1) 
df_train_pca_y = pd.DataFrame(np.append(df_train_pca,trainyield, axis=1),columns = ['A','B','C'])


alt.Chart(df_train_pca_y).mark_circle().encode(x='A',y='B', color='C').configure_view(continuousWidth=200, continuousHeight=150)
alt.show

