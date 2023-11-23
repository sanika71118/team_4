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
#Lets check for null values.
wine.isnull().sum()
# %%
#Lets plot a histogram to visualize data:
import matplotlib.pyplot as plt
import seaborn as sns
wine.hist(bins=20 , figsize= (10,10))
plt.show()


# %%
# 1- Do certain types of wine (red or white) tend to have higher quality scores on average?

average_quality_by_type = wine.groupby('Type')['quality'].mean()
plt.bar(average_quality_by_type.index, average_quality_by_type.values, color=['#C5B4E3', 'lightpink'])
plt.xlabel('Wine Type')
plt.ylabel('Average Quality Score')
plt.title('Average Quality Score by Wine Type')
plt.show()


# %%
# 3- Can we quantitatively measure the correlations between all attributes and wine quality ratings?

attributes = wine.drop(['quality', 'Type'], axis=1) 
quality = wine['quality'] 
correlation_matrix = attributes.corrwith(quality)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='rainbow', cbar=True, fmt=".2f" )
plt.title('Correlation Between Attributes and Wine Quality Ratings')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Attributes')
plt.show()


# %%
# 4- What is the range of wine quality scores, and how can we improve this range through analysis and recommendations?

quality_scores = wine['quality']
plt.figure(figsize=(10, 8))
plt.hist(quality_scores, bins=range(1, 11), edgecolor='black', alpha=0.7, color="lightpink")
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
#%%
#Distribution of Wines by Quality
wine['quality'].value_counts().plot(kind='bar',figsize=(7, 6), rot=0, color="#C5B4E3")
plt.xlabel("Quality")
plt.ylabel("Count of wines")
plt.title("Distribution of Wines by Quality")
plt.show()

#The number of ratings for wine quality '5' and '6' are much larger than that of 3 , 4 and 8.

# %%
#Let's do the correlation matrix for the variables 
plt.figure(figsize = (30,30))
sns.heatmap(wine.corr(),annot=True, cmap= 'gnuplot2')

#From the above heatmap we can conclude that the ‘total sulfur dioxide’ and ‘free sulphur dioxide‘ are highly correlated features

# %%
red_wines = wine[wine['Type'] == 'Red Wine']
white_wines = wine[wine['Type'] == 'White Wine']

# Descriptive statistics for quality scores
print("Red Wine Quality Stats:")
print(red_wines['quality'].describe())
print("\nWhite Wine Quality Stats:")
print(white_wines['quality'].describe())

#%%
plt.figure(figsize=(8, 6))
sns.boxplot(x='Type', y='quality', data=wine, color='lightpink')
plt.xlabel('Wine Type')
plt.ylabel('Quality Score')
plt.title('Quality Scores by Wine Type')
plt.show()
# %%
# Perform a statistical test (e.g., t-test) to compare quality scores

from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(red_wines['quality'], white_wines['quality'])
print(f"T-Statistic: {t_stat}, p-value: {p_value}")

""""
The output of the t-test indicates a strong statistical significance.
The t-statistic of approximately -21.52 suggests a substantial difference in the mean quality scores between red and white wines. Additionally, the p-value, which is significantly smaller than the conventional significance level of 0.05, further supports this difference.
In simple terms, the test results suggest that there is a statistically significant difference in quality scores between red and white wines.

"""

# %%
#2- Can machine learning models accurately predict wine quality based on its chemical composition, and if yes, which algorithms perform the best?
# model development 


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#%%
# make a column which gives best quality of wine as 1 and other as 0
wine['best quality'] = [1 if x > 5 else 0 for x in wine.quality]
wine.head()

# change the Type of wine for white wine to 1 and red wine to 2
wine.replace({'White Wine': 1, 'Red Wine': 0}, inplace=True)
wine.head()
# %%
attributes = wine.drop(['quality', 'best quality'], axis=1) 
quality = wine['best quality'] 
# %%

#the data is split into training and testing with the ratio as 80:20
X_train, X_test, y_train, y_test = train_test_split(attributes, quality, test_size=0.2, random_state=40)
X_train.shape, X_test.shape
# %%

#we will now normalize the data using the Min Max Scaler 
from sklearn.preprocessing import MinMaxScaler
normalize = MinMaxScaler()
X_train = normalize.fit_transform(X_train)
X_test = normalize.transform(X_test)
# %%

# we will now model and fit the data using our Decision Trees 
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predict = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predict)
print(f"Accuracy: {accuracy}")

#%%

## we will now model and fit the data using our Decision Trees 
model2= KNeighborsClassifier()

## Train the model
model2.fit(X_train, y_train)

# Make predictions on the test set
predict2= model2.predict(X_test)

# Evaluate the model0-
accuracy2=accuracy_score(y_test, predict2)
print(f"Accuracy: {accuracy2}")
# %%
