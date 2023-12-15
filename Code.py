#%%
#Import the necessary Libraries:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


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

#%%
#Lets check for null values.
wine.isnull().sum()

# %%
#Let's begin with EDA-

# %%
#Lets plot a histogram to visualize data:
import matplotlib.pyplot as plt
import seaborn as sns
wine.hist(bins=20 , figsize= (10,10), color='#C5B4E3')
plt.show()


# %%
# 1- Do certain types of wine (red or white) tend to have higher quality scores on average?

# calculating the average quality scores of each type
average_quality_by_type = wine.groupby('Type')['quality'].mean()
plt.bar(average_quality_by_type.index, average_quality_by_type.values, color=['#C5B4E3', 'lightpink'])
plt.xlabel('Wine Type')
plt.ylabel('Average Quality Score')
plt.title('Average Quality Score by Wine Type')
plt.show()

#Splitting the dataset into two new types red and white wine:
red_wines = wine[wine['Type'] == 'Red Wine']
white_wines = wine[wine['Type'] == 'White Wine']


#%%
#Correlation Between Attributes and Quality Ratings for Red Wines
red_attributes = red_wines.drop(['quality', 'Type'], axis=1) 
red_quality = red_wines['quality'] 
red_correlation_matrix = red_attributes.corrwith(red_quality)
plt.figure(figsize=(10, 8))
sns.heatmap(red_correlation_matrix.to_frame(), annot=True, cmap='rainbow', cbar=True, fmt=".2f" )
plt.title('Correlation Between Attributes and Red Wine Quality Ratings')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Attributes')
plt.show()

#Correlation Between Attributes and Quality Ratings for White Wines
white_attributes = white_wines.drop(['quality', 'Type'], axis=1) 
white_quality = white_wines['quality'] 
white_correlation_matrix = white_attributes.corrwith(white_quality)
plt.figure(figsize=(10, 8))
sns.heatmap(white_correlation_matrix.to_frame(), annot=True, cmap='rainbow', cbar=True, fmt=".2f" )
plt.title('Correlation Between Attributes and White Wine Quality Ratings')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Attributes')
plt.show()


# %%
# 2- Can we quantitatively measure the correlations between all attributes and wine quality ratings?

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
# 3- What is the range of wine quality scores, and how can we improve this range through analysis and recommendations?

quality_scores = wine['quality']

plt.figure(figsize=(10, 8))
sns.barplot(x=quality_scores.value_counts().index, y=quality_scores.value_counts(), color="lightpink", edgecolor='black', alpha=0.7)
plt.xlabel('Wine Quality Score')
plt.ylabel('Frequency')
plt.title('Distribution of Wine Quality Scores')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#The number of ratings for wine quality '5' and '6' are much larger than that of 3 , 4 and 8.

# %%
#Let's do the correlation matrix for the variables 
plt.figure(figsize = (8,8))
annot_kws = {"size":40}
sns.heatmap(wine.replace({'White Wine': 1, 'Red Wine': 0}).corr(),annot=True, cmap= 'gnuplot2')

#From the above heatmap we can conclude that the ‘total sulfur dioxide’ and ‘free sulphur dioxide‘ are highly correlated to each other
#alcohol is related to quality
#chloride is negatively related to quality
#density is negatively related to quality 

# %%


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
#4- Can machine learning models accurately predict wine quality based on its chemical composition, and if yes, which algorithms perform the best?
# model development 


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
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

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
mean_cv_accuracy = np.mean(cv_scores)
print(f"Mean Cross-validation Accuracy: {mean_cv_accuracy}")

# Make predictions on the test set
predict = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predict)
print(f"Test Accuracy: {accuracy}")
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"Train Accuracy: {train_accuracy}")


# Print & plot feature importance coefficients
print("\nFeature Importance:")
coefficients = model.feature_importances_
for feature, importance in zip(attributes.columns, coefficients):
    print(f"{feature}: {importance}")

abs_coefficients = np.abs(coefficients)
# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(attributes.columns, abs_coefficients, color='skyblue')
plt.xlabel('Absolute Coefficient Values')
plt.title('Feature Importances - Decision Tree')
plt.show()

# Print Confusion Matrix & Classification report
print("\nConfusion Matrix:")
y_pred = model.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
plt.show()
print("\nClassification report:")
print(metrics.classification_report(y_test, model.predict(X_test)))

#%%

## we will now model and fit the data using our KNN 
model2= KNeighborsClassifier()

## Train the model
model2.fit(X_train, y_train)

# Make predictions on the test set
predict2= model2.predict(X_test)

# Evaluate the model
accuracy2=accuracy_score(y_test, predict2)
print(f"Test Accuracy: {accuracy2}")
train_accuracy2 = accuracy_score(y_train, model2.predict(X_train))
print(f"Train Accuracy: {train_accuracy2}")

print("\nConfusion Matrix:")
y_pred = model2.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
plt.show()
print("\nClassification report:")
print(metrics.classification_report(y_test, model2.predict(X_test)))

# %%

## we will now model and fit the data using our Logistic Regression

from sklearn.linear_model import LogisticRegression

# Logistic Regression
model3 = LogisticRegression()

# Train the model
model3.fit(X_train, y_train)

# Make predictions on the test set
predict3 = model3.predict(X_test)

# Evaluate the model
accuracy3 = accuracy_score(y_test, predict3)
print(f"Test Accuracy: {accuracy3}")

# Perform 5-fold cross-validation during training
cv_scores3 = cross_val_score(model3, X_train, y_train, cv=5)
mean_cv_accuracy3 = np.mean(cv_scores3)
print(f"Train Accuracy: {mean_cv_accuracy3}")


# Print feature importance coefficients
coefficients2 = model3.coef_[0]
print("\nFeature Importance:")
for feature, importance in zip(attributes.columns, coefficients2):
    print(f"{feature}: {importance}")

# Create a bar plot of the feature importance
abs_coefficients2 = np.abs(coefficients2)
plt.figure(figsize=(10, 6))
plt.barh(attributes.columns, abs_coefficients2, color='skyblue')
plt.xlabel('Absolute Coefficient Values')
plt.title('Feature Importances - Logistic Regression')
plt.show()

print("\nConfusion Matrix:")
y_pred = model3.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
plt.show()
print("\nClassification report:")
print(metrics.classification_report(y_test, model3.predict(X_test)))


# %%[markdown]

# Part - 2

### Considering only the top 4 correlated features
#### Features -> Alcohol, density, chlorides, volatile acidity



#%%

## 'alcohol' vs 'quality' plot
plt.figure(figsize=(10, 8))

# Create a box plot
sns.boxplot(x='quality', y='alcohol', data=wine, palette='viridis')
plt.xlabel('Wine Quality')
plt.ylabel('Alcohol Content')
plt.title('Box Plot of Wine Quality and Alcohol Content')

# Show the plot
plt.show()


#%%

## 'density' vs 'quality' plot

# Violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='quality', y='density', data=wine, hue='Type', split=True, palette='viridis')

# Set plot title and labels
plt.title('Distribution of Wine Quality Across Density Levels')
plt.xlabel('Quality')
plt.ylabel('Density')

# Display the plot
plt.show()


#%%

## 'volatile acidity' vs 'quality' plot

# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='volatile acidity', data=wine, hue='Type', palette='viridis')

# Set plot title and labels
plt.title('Average Volatile Acidity Across Wine Quality Levels')
plt.xlabel('Quality')
plt.ylabel('Average Volatile Acidity')

# Display the plot
plt.show()


# %%
new_attributes = wine.drop(['quality', 'best quality', 'fixed acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates'], axis=1) 
quality = wine['best quality'] 

# %%

#the data is split into training and testing with the ratio as 80:20
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_attributes, quality, test_size=0.2, random_state=40)
X_train_new.shape, X_test_new.shape

# %%

#we will now normalize the data using the Min Max Scaler 
from sklearn.preprocessing import MinMaxScaler
normalize = MinMaxScaler()
X_train_new = normalize.fit_transform(X_train_new)
X_test_new = normalize.transform(X_test_new)
# %%

# we will now model and fit the data using our Decision Trees 
model_new = DecisionTreeClassifier()

# Train the model
model_new.fit(X_train_new, y_train_new)

# Make predictions on the test set
predict_new = model_new.predict(X_test_new)

# Evaluate the model
accuracy_new = accuracy_score(y_test_new, predict_new)
print(f"Accuracy: {accuracy_new}")

#%%

## we will now model and fit the data using our Decision Trees 
model2_new= KNeighborsClassifier()

## Train the model
model2_new.fit(X_train_new, y_train_new)

# Make predictions on the test set
predict2_new= model2_new.predict(X_test_new)

# Evaluate the model0-
accuracy2_new=accuracy_score(y_test_new, predict2_new)
print(f"Accuracy: {accuracy2_new}")
# %%


from sklearn.linear_model import LogisticRegression

# Logistic Regression
model3_new = LogisticRegression()

# Train the model
model3_new.fit(X_train_new, y_train_new)

# Make predictions on the test set
predict3_new = model3_new.predict(X_test_new)

# Evaluate the model
accuracy3_new = accuracy_score(y_test_new, predict3_new)
print(f"Accuracy: {accuracy3_new}")

# %%
#model validation
models_new=[model_new, model2_new, model3_new ]

#%%

#decision trees
y_pred_new = models_new[0].predict(X_test_new)
print(metrics.confusion_matrix(y_test_new, y_pred_new))
print(metrics.classification_report(y_test_new, models_new[0].predict(X_test_new)))

plt.show()
# %%
#knn
y_pred_new = models_new[1].predict(X_test_new)
print(metrics.confusion_matrix(y_test_new, y_pred_new))
print(metrics.classification_report(y_test_new, models_new[1].predict(X_test_new)))

plt.show()

#%%

#logistic regression

y_pred_new = models_new[2].predict(X_test_new)
print(metrics.confusion_matrix(y_test_new, y_pred_new))
print(metrics.classification_report(y_test_new, models_new[2].predict(X_test_new)))

plt.show()


#%%

### Regression Models

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# %%

# Standardize the data
scaler = StandardScaler()
X_train_sscaled = scaler.fit_transform(X_train)
X_test_sscaled = scaler.transform(X_test)

# %%

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_sscaled, y_train)
# Make predictions on the test set
linear_predict = linear_model.predict(X_test_sscaled)

# Evaluate Linear Regression
linear_mse = mean_squared_error(y_test, linear_predict)
linear_r2 = r2_score(y_test, linear_predict)
print("Linear Regression:")
print(f"Mean Squared Error: {linear_mse}")
print(f"R-squared: {linear_r2}\n")

# %%

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train_sscaled, y_train)

# Make predictions on the test set
lasso_predict = lasso_model.predict(X_test_sscaled)

# Evaluate Lasso Regression
lasso_mse = mean_squared_error(y_test, lasso_predict)
lasso_r2 = r2_score(y_test, lasso_predict)
print("Lasso Regression:")
print(f"Mean Squared Error: {lasso_mse}")
print(f"R-squared: {lasso_r2}\n")

# %%

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train_sscaled, y_train)

# Make predictions on the test set
ridge_predict = ridge_model.predict(X_test_sscaled)

# Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_predict)
ridge_r2 = r2_score(y_test, ridge_predict)
print("Ridge Regression:")
print(f"Mean Squared Error: {ridge_mse}")
print(f"R-squared: {ridge_r2}")

# %%
