#%%
#Import the necessary Libraries:
import os
import numpy as np
import pandas as pd

# %%
#To read the input file:
wine = pd.read_csv('WineQuality.csv')
wine.head()

# %%
# What are the columns in the dataset?
column_names = wine.columns
print("The variables in the dataset are : \n",column_names)

#%%
# What are the datatypes of each column in the dataset ?
datatypes = wine.info()

# %%
#Dropping the first column in the dataset
wine = wine.drop(wine.columns[0], axis=1)
wine.head()

#%%
#Descriptive statistics of the dataset
wine.describe().T

# %%
#Let's begin with EDA-

