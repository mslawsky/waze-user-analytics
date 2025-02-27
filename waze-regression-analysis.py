#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 5 - Regression analysis: Simplify complex data relationships**

# Your team is more than halfway through their user churn project. Earlier, you completed a project proposal, used Python to explore and analyze Waze’s user data, created data visualizations, and conducted a hypothesis test. Now, leadership wants your team to build a regression model to predict user churn based on a variety of variables.
# 
# You check your inbox and discover a new email from Ursula Sayo, Waze's Operations Manager. Ursula asks your team about the details of the regression model. You also notice two follow-up emails from your supervisor, May Santner. The first email is a response to Ursula, and says that the team will build a binomial logistic regression model. In her second email, May asks you to help build the model and prepare an executive summary to share your results.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions and prepare an executive summary.

# # **Course 5 End-of-course project: Regression modeling**
# 
# In this activity, you will build a binomial logistic regression model. As you have learned, logistic regression helps you estimate the probability of an outcome. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed.
# <br/>
# 
# **The purpose** of this project is to demostrate knowledge of exploratory data analysis (EDA) and a binomial logistic regression model.
# 
# **The goal** is to build a binomial logistic regression model and evaluate the model's performance.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** EDA & Checking Model Assumptions
# * What are some purposes of EDA before constructing a binomial logistic regression model?
# 
# **Part 2:** Model Building and Evaluation
# * What resources do you find yourself using as you complete this stage?
# 
# **Part 3:** Interpreting Model Results
# 
# * What key insights emerged from your model(s)?
# 
# * What business recommendations do you propose based on the models built?
# 
# <br/>
# 
# Follow the instructions and answer the question below to complete the activity. Then, you will complete an executive summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.

# # **Build a regression model**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.

# ### **Task 1. Imports and data loading**
# Import the data and packages that you've learned are needed for building logistic regression models.

# In[1]:


# Packages for numerics + dataframes
import numpy as np
import pandas as pd

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for Logistic Regression & Confusion Matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# Import the dataset.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load the dataset by running this cell

df = pd.read_csv('waze_dataset.csv')


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.
# 
# In this stage, consider the following question:
# 
# * What are some purposes of EDA before constructing a binomial logistic regression model?

# The purposes of EDA before constructing a binomial logistic regression model include:
# 
# -Understanding the distribution of the target variable (churn vs. retained) to assess class imbalance
# -Identifying potential relationships between predictors and the target variable
# Detecting outliers that might influence model performance
# -Checking for missing values that need to be addressed
# -Assessing multicollinearity among predictors
# -Determining which variables should be included in the model
# -Identifying potential data transformations needed for certain variables
# -Examining if the assumptions of logistic regression are satisfied
# -Gaining domain knowledge insights to better interpret model results

# ### **Task 2a. Explore data with EDA**
# 
# Analyze and discover data, looking for correlations, missing data, potential outliers, and/or duplicates.
# 
# 

# Start with `.shape` and `info()`.

# In[4]:


print(df.shape)
df.info()

print("\nMissing values per column:")
print(df.isna().sum())


# **Question:** Are there any missing values in your data?

# Yes, there are missing values in the dataset. Specifically, there are 700 missing values in the 'label' column, which is our target variable. All other columns have complete data with no missing values.
# 
# Since the missing values are only in the target variable and represent less than 5% of the total data, the recommended approach is to drop these rows rather than attempt to impute values. This is especially appropriate when dealing with a classification task like predicting churn, as imputing target values could introduce bias into the model.

# Use `.head()`.
# 
# 

# In[5]:


df.head()


# Use `.drop()` to remove the ID column since we don't need this information for your analysis.

# In[6]:


df = df.drop('ID', axis=1)


# Now, check the class balance of the dependent (target) variable, `label`.

# In[7]:


print("\nClass distribution of 'label':")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))


# Call `.describe()` on the data.
# 

# In[8]:


df.describe()


# **Question:** Are there any variables that could potentially have outliers just by assessing at the quartile values, standard deviation, and max values?

# Based on the descriptive statistics, several variables show strong indications of having outliers:
# 
# 1.sessions: The maximum value (743) is more than 6 times the 75th percentile (112), with a high standard deviation relative to the mean.
# 2.drives: The maximum value (596) is more than 6 times the 75th percentile (93).
# 3.total_sessions: The maximum value (1216.15) is about 4.8 times higher than the 75th percentile (254.19).
# 4.total_navigations_fav1: The maximum value (1236) is almost 7 times higher than the 75th percentile (178).
# 5.total_navigations_fav2: The maximum value (415) is nearly 10 times higher than the 75th percentile (43).
# 6.driven_km_drives: The maximum value (21183.40) is about 4 times higher than the 75th percentile (5289.86).
# 7.duration_minutes_drives: The maximum value (15851.73) is more than 6 times higher than the 75th percentile (2464.36).
# 
# These variables all show right-skewed distributions with a few very high values far above the typical range for most users. In the modeling approach, these outliers will be handled by imputing values above the 95th percentile to the 95th percentile value, which is a common technique to reduce the influence of extreme values while preserving the overall data distribution.

# ### **Task 2b. Create features**
# 
# Create features that may be of interest to the stakeholder and/or that are needed to address the business scenario/problem.

# #### **`km_per_driving_day`**
# 
# You know from earlier EDA that churn rate correlates with distance driven per driving day in the last month. It might be helpful to engineer a feature that captures this information.
# 
# 1. Create a new column in `df` called `km_per_driving_day`, which represents the mean distance driven per driving day for each user.
# 
# 2. Call the `describe()` method on the new column.

# In[9]:


# 1. Create `km_per_driving_day` column
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# Check descriptive stats of new column
df['km_per_driving_day'].describe()


# Note that some values are infinite. This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.
# 
# 1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.
# 
# 2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.

# In[10]:


# 1. Convert infinite values to zero
df['km_per_driving_day'] = df['km_per_driving_day'].replace(np.inf, 0)

# Confirm the conversion worked
df['km_per_driving_day'].describe()


# #### **`professional_driver`**
# 
# Create a new, binary feature called `professional_driver` that is a 1 for users who had 60 or more drives <u>**and**</u> drove on 15+ days in the last month.
# 
# **Note:** The objective is to create a new feature that separates professional drivers from other drivers. In this scenario, domain knowledge and intuition are used to determine these deciding thresholds, but ultimately they are arbitrary.

# To create this column, use the [`np.where()`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function. This function accepts as arguments:
# 1. A condition
# 2. What to return when the condition is true
# 3. What to return when the condition is false
# 
# ```
# Example:
# x = [1, 2, 3]
# x = np.where(x > 2, 100, 0)
# x
# array([  0,   0, 100])
# ```

# In[11]:


# Create `professional_driver` column
df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)


# Perform a quick inspection of the new variable.
# 
# 1. Check the count of professional drivers and non-professionals
# 
# 2. Within each class (professional and non-professional) calculate the churn rate

# In[12]:


# 1. Check count of professionals and non-professionals
print("\nCount of professional drivers:")
print(df['professional_driver'].value_counts())


# The churn rate for professional drivers is 7.6%, while the churn rate for non-professionals is 19.9%. This seems like it could add predictive signal to the model.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# After analysis and deriving variables with close relationships, it is time to begin constructing the model.
# 
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.
# 
# In this stage, consider the following question:
# 
# * Why did you select the X variables you did?

# For the model, we exclude:
# 
# -'label' and 'label2' as they are the target variables
# -'device' as we created a binary-encoded version 'device2'
# -'sessions' due to high multicollinearity with 'drives'
# -'driving_days' due to high multicollinearity with 'activity_days'
# 
# We retain the other variables because they provide predictive signal for churn while minimizing multicollinearity issues. The variables 'drives' and 'activity_days' were kept over their multicollinear counterparts because they have slightly stronger correlations with the target variable.

# ### **Task 3a. Preparing variables**

# Call `info()` on the dataframe to check the data type of the `label` variable and to verify if there are any missing values.

# In[13]:


df.info()


# Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.

# In[15]:


# Drop rows with missing data in `label` column
df = df.dropna(subset=['label'])


# #### **Impute outliers**
# 
# You rarely want to drop outliers, and generally will not do so unless there is a clear reason for it (e.g., typographic errors).
# 
# At times outliers can be changed to the **median, mean, 95th percentile, etc.**
# 
# Previously, you determined that seven of the variables had clear signs of containing outliers:
# 
# * `sessions`
# * `drives`
# * `total_sessions`
# * `total_navigations_fav1`
# * `total_navigations_fav2`
# * `driven_km_drives`
# * `duration_minutes_drives`
# 
# For this analysis, impute the outlying values for these columns. Calculate the **95th percentile** of each column and change to this value any value in the column that exceeds it.
# 

# In[16]:


# Impute outliers
outlier_columns = ['sessions', 'drives', 'total_sessions', 'total_navigations_fav1', 
                  'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives']

for col in outlier_columns:
    threshold = df[col].quantile(0.95)
    df[col] = np.where(df[col] > threshold, threshold, df[col])


# Call `describe()`.

# In[17]:


df.describe()


# #### **Encode categorical variables**

# Change the data type of the `label` column to be binary. This change is needed to train a logistic regression model.
# 
# Assign a `0` for all `retained` users.
# 
# Assign a `1` for all `churned` users.
# 
# Save this variable as `label2` as to not overwrite the original `label` variable.
# 
# **Note:** There are many ways to do this. Consider using `np.where()` as you did earlier in this notebook.

# In[18]:


# Create binary `label2` column
df['label2'] = np.where(df['label'] == 'churned', 1, 0)


# ### **Task 3b. Determine whether assumptions have been met**
# 
# The following are the assumptions for logistic regression:
# 
# * Independent observations (This refers to how the data was collected.)
# 
# * No extreme outliers
# 
# * Little to no multicollinearity among X predictors
# 
# * Linear relationship between X and the **logit** of y
# 
# For the first assumption, you can assume that observations are independent for this project.
# 
# The second assumption has already been addressed.
# 
# The last assumption will be verified after modeling.
# 
# **Note:** In practice, modeling assumptions are often violated, and depending on the specifics of your use case and the severity of the violation, it might not affect your model much at all or it will result in a failed model.

# #### **Collinearity**
# 
# Check the correlation among predictor variables. First, generate a correlation matrix.

# In[19]:


# Generate a correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)


# Now, plot a correlation heatmap.

# In[20]:


# Plot correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# If there are predictor variables that have a Pearson correlation coefficient value greater than the **absolute value of 0.7**, these variables are strongly multicollinear. Therefore, only one of these variables should be used in your model.
# 
# **Note:** 0.7 is an arbitrary threshold. Some industries may use 0.6, 0.8, etc.
# 
# **Question:** Which variables are multicollinear with each other?

# Based on the correlation heatmap, the following pairs of variables show strong multicollinearity (correlation coefficient > 0.7):
# 
# 1.sessions and drives (correlation = 1.00): These variables are perfectly correlated, suggesting they capture essentially the same information about user behavior.
# 2.activity_days and driving_days (correlation = 0.95): These variables have a very strong positive correlation, indicating that the number of days a user is active closely tracks with the number of days they drive.
# 3.driven_km_drives and duration_minutes_drives (correlation = 0.69): While just below the 0.7 threshold, this correlation is still quite high and worth noting. This suggests that the distance driven and time spent driving are closely related, as expected.

# ### **Task 3c. Create dummies (if necessary)**
# 
# If you have selected `device` as an X variable, you will need to create dummy variables since this variable is categorical.
# 
# In cases with many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.
# 
# **Note:** Variables with many categories should only be dummied if absolutely necessary. Each category will result in a coefficient for your model which can lead to overfitting.
# 
# Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.
# 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# In[21]:


# Create new `device2` variable
df['device2'] = np.where(df['device'] == 'iPhone', 1, 0)


# ### **Task 3d. Model building**

# #### **Assign predictor variables and target**
# 
# To build your model you need to determine what X variables you want to include in your model to predict your target&mdash;`label2`.
# 
# Drop the following variables and assign the results to `X`:
# 
# * `label` (this is the target)
# * `label2` (this is the target)
# * `device` (this is the non-binary-encoded categorical variable)
# * `sessions` (this had high multicollinearity)
# * `driving_days` (this had high multicollinearity)
# 
# **Note:** Notice that `sessions` and `driving_days` were selected to be dropped, rather than `drives` and `activity_days`. The reason for this is that the features that were kept for modeling had slightly stronger correlations with the target variable than the features that were dropped.

# In[22]:


# Isolate predictor variables
X = df.drop(['label', 'label2', 'device', 'sessions', 'driving_days'], axis=1)


# Now, isolate the dependent (target) variable. Assign it to a variable called `y`.

# In[23]:


# Isolate target variable
y = df['label2']


# #### **Split the data**
# 
# Use scikit-learn's [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to perform a train/test split on your data using the X and y variables you assigned above.
# 
# **Note 1:** It is important to do a train test to obtain accurate predictions.  You always want to fit your model on your training set and evaluate your model on your test set to avoid data leakage.
# 
# **Note 2:** Because the target class is imbalanced (82% retained vs. 18% churned), you want to make sure that you don't get an unlucky split that over- or under-represents the frequency of the minority class. Set the function's `stratify` parameter to `y` to ensure that the minority class appears in both train and test sets in the same proportion that it does in the overall dataset.

# In[24]:


# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[25]:


# Use .head()
X_train.head()


# Use scikit-learn to instantiate a logistic regression model. Add the argument `penalty = None`.
# 
# It is important to add `penalty = None` since your predictors are unscaled.
# 
# Refer to scikit-learn's [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) documentation for more information.
# 
# Fit the model on `X_train` and `y_train`.

# In[29]:


log_model = LogisticRegression(penalty='l2', C=1e6, random_state=42, 
                              solver='liblinear', multi_class='ovr')
log_model.fit(X_train, y_train)


# Call the `.coef_` attribute on the model to get the coefficients of each variable.  The coefficients are in order of how the variables are listed in the dataset.  Remember that the coefficients represent the change in the **log odds** of the target variable for **every one unit increase in X**.
# 
# If you want, create a series whose index is the column names and whose values are the coefficients in `model.coef_`.

# In[30]:


coefficients = pd.Series(log_model.coef_[0], index=X_train.columns)
print("\nModel Coefficients:")
print(coefficients)


# Call the model's `intercept_` attribute to get the intercept of the model.

# In[31]:


print("\nModel Intercept:")
print(log_model.intercept_)


# #### **Check final assumption**
# 
# Verify the linear relationship between X and the estimated log odds (known as logits) by making a regplot.
# 
# Call the model's `predict_proba()` method to generate the probability of response for each sample in the training data. (The training data is the argument to the method.) Assign the result to a variable called `training_probabilities`. This results in a 2-D array where each row represents a user in `X_train`. The first column is the probability of the user not churning, and the second column is the probability of the user churning.

# In[32]:


# Get the predicted probabilities of the training data
training_probabilities = log_model.predict_proba(X_train)


# In logistic regression, the relationship between a predictor variable and the dependent variable does not need to be linear, however, the log-odds (a.k.a., logit) of the dependent variable with respect to the predictor variable should be linear. Here is the formula for calculating log-odds, where _p_ is the probability of response:
# <br>
# $$
# logit(p) = ln(\frac{p}{1-p})
# $$
# <br>
# 
# 1. Create a dataframe called `logit_data` that is a copy of `df`.
# 
# 2. Create a new column called `logit` in the `logit_data` dataframe. The data in this column should represent the logit for each user.
# 

# In[33]:


# 1. Copy the `X_train` dataframe and assign to `logit_data`
logit_data = X_train.copy()

# Create logit column
logit_data['logit'] = np.log(training_probabilities[:, 1] / (1 - training_probabilities[:, 1]))


# Plot a regplot where the x-axis represents an independent variable and the y-axis represents the log-odds of the predicted probabilities.
# 
# In an exhaustive analysis, this would be plotted for each continuous or discrete predictor variable. Here we show only `driving_days`.

# In[34]:


# Plot regplot of `activity_days` log-odds
plt.figure(figsize=(10, 6))
sns.regplot(x='activity_days', y='logit', data=logit_data)
plt.title('Log-Odds vs Activity Days')
plt.xlabel('Activity Days')
plt.ylabel('Log-Odds (Logit)')
plt.show()


# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 4a. Results and evaluation**
# 
# If the logistic assumptions are met, the model results can be appropriately interpreted.
# 
# Use the code block below to make predictions on the test data.
# 

# In[35]:


# Generate predictions on X_test
y_preds = log_model.predict(X_test)


# Now, use the `score()` method on the model with `X_test` and `y_test` as its two arguments. The default score in scikit-learn is **accuracy**.  What is the accuracy of your model?
# 
# *Consider:  Is accuracy the best metric to use to evaluate this model?*

# In[36]:


# Score the model (accuracy) on the test data
accuracy = log_model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")


# ### **Task 4b. Show results with a confusion matrix**

# Use the `confusion_matrix` function to obtain a confusion matrix. Use `y_test` and `y_preds` as arguments.

# In[37]:


cm = confusion_matrix(y_test, y_preds)
print("\nConfusion Matrix:")
print(cm)


# Next, use the `ConfusionMatrixDisplay()` function to display the confusion matrix from the above cell, passing the confusion matrix you just created as its argument.

# In[38]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retained', 'Churned'])
plt.figure(figsize=(8, 6))
disp.plot()
plt.title('Confusion Matrix')
plt.show()


# You can use the confusion matrix to compute precision and recall manually. You can also use scikit-learn's [`classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) function to generate a table from `y_test` and `y_preds`.

# In[39]:


# Calculate precision manually
true_positives = cm[1][1]
false_positives = cm[0][1]
precision = true_positives / (true_positives + false_positives)
print(f"\nPrecision (manually calculated): {precision:.4f}")


# In[40]:


# Calculate recall manually
false_negatives = cm[1][0]
recall = true_positives / (true_positives + false_negatives)
print(f"\nRecall (manually calculated): {recall:.4f}")


# In[41]:


# Create a classification report
class_report = classification_report(y_test, y_preds, target_names=['Retained', 'Churned'])
print("\nClassification Report:")
print(class_report)


# **Note:** The model has decent precision but very low recall, which means that it makes a lot of false negative predictions and fails to capture users who will churn.

# ### **BONUS**
# 
# Generate a bar graph of the model's coefficients for a visual representation of the importance of the model's features.

# In[42]:


# Create a list of (column_name, coefficient) tuples
coef_importance = list(zip(X_train.columns, log_model.coef_[0]))
coef_importance.sort(key=lambda x: abs(x[1]), reverse=True)

feature_names = [item[0] for item in coef_importance]
feature_coefficients = [item[1] for item in coef_importance]

plt.figure(figsize=(12, 8))
bars = plt.barh(feature_names, feature_coefficients)

# Sort the list by coefficient value
for i, bar in enumerate(bars):
    if feature_coefficients[i] < 0:
        bar.set_color('red')
    else:
        bar.set_color('blue')


# In[43]:


# Plot the feature importances
plt.title('Feature Importance (Model Coefficients)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()


# ### **Task 4c. Conclusion**
# 
# Now that you've built your regression model, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. What variable most influenced the model's prediction? How? Was this surprising?
# 
# 2. Were there any variables that you expected to be stronger predictors than they were?
# 
# 3. Why might a variable you thought to be important not be important in the model?
# 
# 4. Would you recommend that Waze use this model? Why or why not?
# 
# 5. What could you do to improve this model?
# 
# 6. What additional features would you like to have to help improve the model?
# 

# 1.What variable most influenced the model's prediction? How? Was this surprising?
# The variable that most influenced the model's prediction was activity_days with a coefficient of -0.103572. This strong negative coefficient indicates that users who open the app more frequently are significantly less likely to churn. This isn't particularly surprising as engagement is typically a strong predictor of retention across digital products. However, the magnitude of this effect compared to other variables highlights just how crucial regular app usage is for retention.
# 
# 2.Were there any variables that you expected to be stronger predictors than they were?
# The professional_driver variable had a relatively modest impact (-0.001565) despite the substantial difference in churn rates between professional (7.6%) and non-professional (19.9%) drivers observed in the EDA. Similarly, drives and driven_km_drives had smaller impacts than expected, suggesting that it's the frequency of app usage (opening the app) rather than actual driving that most influences retention.
# 
# 3.Why might a variable you thought to be important not be important in the model?
# 
# Variables might appear less important in the model due to:
# 
# -Multicollinearity with other variables (e.g., professional_driver capturing similar information as activity_days)
# -The relationship being captured better by engineered features
# -The relationship being non-linear and not well-captured by the logistic regression model
# -The true relationship being more complex and requiring interaction terms
# 
# 
# 4.Would you recommend that Waze use this model? Why or why not?
# The model provides a useful starting point but should not be deployed as-is. With an accuracy of 82.55%, it appears reasonably good at first glance. However, the low recall (9.66%) means it misses about 90% of users who will actually churn. For a churn prediction model, this is problematic as the whole point is to identify users at risk. I would recommend using the insights from this model to inform retention strategies while developing a more sophisticated model with better recall.
# 
# 5.What could you do to improve this model?
# 
# To improve the model:
# 
# -Test alternative algorithms (Random Forest, Gradient Boosting, Neural Networks)
# -Address class imbalance through techniques like SMOTE or class weighting
# -Try more sophisticated feature engineering
# -Explore interaction terms between features
# -Implement cross-validation for better hyperparameter tuning
# -Consider ensemble methods to combine multiple models
# -Evaluate different probability thresholds for classification to optimize the precision-recall tradeoff
# 
# 
# 6.What additional features would you like to have to help improve the model?
# Additional helpful features would include:
# 
# User satisfaction metrics (ratings, feedback)
# -App performance data (crashes, errors, load times)
# -Feature usage patterns (which Waze features users engage with)
# -User demographic information
# -Time-based features (time since last use, usage trends over time)
# -Seasonal or temporal pattern data
# -Competitor app usage information if available
# -User-reported issues or complaints
# -Device-specific performance metrics

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 
