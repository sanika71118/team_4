#%%
#Import the necessary Libraries:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
#To read the input file:
wine = pd.read_csv('WineQuality.csv')
wine.head()

# %%
# What are the columns in the dataset?
column_names = wine.columns
print("The variables in the dataset are : \n",column_names)

#%%
# What are the datatypes of each column in the dataset?
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


# %%
# 1- Do certain types of wine (red or white) tend to have higher quality scores on average?

average_quality_by_type = wine.groupby('Type')['quality'].mean()
plt.bar(average_quality_by_type.index, average_quality_by_type.values, color=['red', 'blue'])
plt.xlabel('Wine Type')
plt.ylabel('Average Quality Score')
plt.title('Average Quality Score by Wine Type')
plt.show()

# %%
# 2- Can machine learning models accurately predict wine quality based on its chemical composition, and if yes, which algorithms perform the best?

# %%
# 3- Can we quantitatively measure the correlations between all attributes and wine quality ratings?

attributes = wine.drop(['quality', 'Type'], axis=1) 
quality = wine['quality'] 
correlation_matrix = attributes.corrwith(quality)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', cbar=True, fmt=".2f" )
plt.title('Correlation Between Attributes and Wine Quality Ratings')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Attributes')
plt.show()


# %%
# 4- What is the range of wine quality scores, and how can we improve this range through analysis and recommendations?

quality_scores = wine['quality']
plt.figure(figsize=(10, 8))
plt.hist(quality_scores, bins=range(1, 11), edgecolor='black', alpha=0.7)
plt.xlabel('Wine Quality Score')
plt.ylabel('Frequency')
plt.title('Distribution of Wine Quality Scores')
plt.xticks(range(1, 11)) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# %%
# 5- How will understanding and improving wine quality benefit winemakers, distributors, and wine consumers?

# %%
#
