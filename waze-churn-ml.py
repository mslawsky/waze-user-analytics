#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 6 - The nuts and bolts of machine learning**

# Your team is close to completing their user churn project. Previously, you completed a project proposal, and used Python to explore and analyze Waze’s user data, create data visualizations, and conduct a hypothesis test. Most recently, you built a binomial logistic regression model based on multiple variables.
# 
# Leadership appreciates all your hard work. Now, they want your team to build a machine learning model to predict user churn. To get the best results, your team decides to build and test two tree-based models: random forest and XGBoost.
# 
# Your work will help leadership make informed business decisions to prevent user churn, improve user retention, and grow Waze’s business.
# 

# # **Course 6 End-of-Course Project: Build a machine learning model**
# 
# In this activity, you will practice using tree-based modeling techniques to predict on a binary target class.
# <br/>
# 
# **The purpose** of this model is to find factors that drive user churn.
# 
# **The goal** of this model is to predict whether or not a Waze user is retained or churned.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Build a machine learning model**
# 

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 1.   What are you being asked to do?
# 
# 
# 2.   What are the ethical implications of the model? What are the consequences of your model making errors?
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a Waze user won't churn, but they actually will)?
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a Waze user will churn, but they actually won't)?
# 
# 3.  Do the benefits of such a model outweigh the potential problems?
# 4.  Would you proceed with the request to build this model? Why or why not?
# 
# 

# 1-4. Ethics and Model Implications:
# The model aims to predict user churn at Waze. Let's consider the ethical implications:
# 
# 1.A false negative (predicting a user won't churn when they actually will) means Waze might miss an opportunity to intervene and retain the user. This could lead to unnecessary user loss and reduced platform engagement.
# 2.A false positive (predicting a user will churn when they actually won't) might lead to unnecessary retention efforts directed at satisfied users. This could waste resources but has minimal negative impact on users - perhaps just receiving unneeded engagement incentives.
# 3.The benefits outweigh the potential problems. The model helps Waze identify at-risk users and take proactive steps to improve user experience and retention. The consequences of errors are relatively low-impact with no harm to users.
# 4.I would proceed with building this model. The potential upside of improved user retention far outweighs the minimal downsides. There are no significant ethical concerns, as the model informs business decisions that benefit both users (better experience) and Waze (higher retention).

# ### **Task 1. Imports and data loading**
# 
# Import packages and libraries needed to build and evaluate random forest and XGBoost classification models.

# In[1]:


# Import packages for data manipulation
import numpy as np
import pandas as pd

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)

# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as XGBClassifier

# This is the function that helps plot feature importance
from xgboost import plot_importance

# # This module lets us save our models once we fit them.
import pickle


# Now read in the dataset as `df0` and inspect the first five rows.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Import dataset
df0 = pd.read_csv('waze_dataset.csv')


# In[3]:


# Inspect the first five rows
df0.head()


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2. Feature engineering**
# 
# You have already prepared much of this data and performed exploratory data analysis (EDA) in previous courses. You know that some features had stronger correlations with churn than others, and you also created some features that may be useful.
# 
# In this part of the project, you'll engineer these features and some new features to use for modeling.
# 
# To begin, create a copy of `df0` to preserve the original dataframe. Call the copy `df`.

# In[4]:


# Copy the df0 dataframe
df = df0.copy()


# Call `info()` on the new dataframe so the existing columns can be easily referenced.

# In[5]:


df.info()


# #### **`km_per_driving_day`**
# 
# 1. Create a feature representing the mean number of kilometers driven on each driving day in the last month for each user. Add this feature as a column to `df`.
# 
# 2. Get descriptive statistics for this new feature
# 
# 

# In[6]:


# 1. Create `km_per_driving_day` feature
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Get descriptive stats
df['km_per_driving_day'].describe()


# Notice that some values are infinite. This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.
# 
# 1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.
# 
# 2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.

# In[7]:


# 1. Convert infinite values to zero
df['km_per_driving_day'] = df['km_per_driving_day'].replace(np.inf, 0)

# 2. Confirm that it worked
df['km_per_driving_day'].describe()


# #### **`percent_sessions_in_last_month`**
# 
# 1. Create a new column `percent_sessions_in_last_month` that represents the percentage of each user's total sessions that were logged in their last month of use.
# 
# 2. Get descriptive statistics for this new feature

# In[8]:


# 1. Create `percent_sessions_in_last_month` feature
df['percent_sessions_in_last_month'] = (df['sessions'] / df['total_sessions']) * 100


# 1. Get descriptive stats
df['percent_sessions_in_last_month'].describe()


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

# In[9]:


# Create `professional_driver` feature
df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)


# #### **`total_sessions_per_day`**
# 
# Now, create a new column that represents the mean number of sessions per day _since onboarding_.

# In[11]:


# Create `total_sessions_per_day` feature
df['total_sessions_per_day'] = df['total_sessions'] / df['n_days_after_onboarding']


# As with other features, get descriptive statistics for this new feature.

# In[12]:


# Get descriptive stats
df['total_sessions_per_day'].describe()


# #### **`km_per_hour`**
# 
# Create a column representing the mean kilometers per hour driven in the last month.

# In[13]:


# Create `km_per_hour` feature
df['km_per_hour'] = df['driven_km_drives'] / df['duration_minutes_drives'] * 60


# #### **`km_per_drive`**
# 
# Create a column representing the mean number of kilometers per drive made in the last month for each user. Then, print descriptive statistics for the feature.

# In[14]:


# Create `km_per_drive` feature
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
df['km_per_drive'].describe()


# This feature has infinite values too. Convert the infinite values to zero, then confirm that it worked.

# In[15]:


# 1. Convert infinite values to zero
df['km_per_drive'] = df['km_per_drive'].replace(np.inf, 0)

# 2. Confirm that it worked
df['km_per_drive'].describe()


# #### **`percent_of_sessions_to_favorite`**
# 
# Finally, create a new column that represents the percentage of total sessions that were used to navigate to one of the users' favorite places. Then, print descriptive statistics for the new column.
# 
# This is a proxy representation for the percent of overall drives that are to a favorite place. Since total drives since onboarding are not contained in this dataset, total sessions must serve as a reasonable approximation.
# 
# People whose drives to non-favorite places make up a higher percentage of their total drives might be less likely to churn, since they're making more drives to less familiar places.

# In[16]:


# Create `percent_of_sessions_to_favorite` feature
df['percent_of_sessions_to_favorite'] = (df['n_days_after_onboarding'] / df['total_sessions']) * 100


# Get descriptive stats
df['percent_of_sessions_to_favorite'].describe()


# ### **Task 3. Drop missing values**
# 
# Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.

# In[17]:


# Drop rows with missing values
df = df.dropna()


# ### **Task 4. Outliers**
# 
# You know from previous EDA that many of these columns have outliers. However, tree-based models are resilient to outliers, so there is no need to make any imputations.

# ### **Task 5. Variable encoding**

# #### **Dummying features**
# 
# In order to use `device` as an X variable, you will need to convert it to binary, since this variable is categorical.
# 
# In cases where the data contains many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.
# 
# **Note:** Each possible category of each feature will result in a feature for your model, which could lead to an inadequate ratio of features to observations and/or difficulty understanding your model's predictions.
# 
# Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.
# 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# In[18]:


# Create new `device2` variable
df['device2'] = np.where(df['device'] == 'iPhone', 1, 0)


# #### **Target encoding**
# 
# The target variable is also categorical, since a user is labeled as either "churned" or "retained." Change the data type of the `label` column to be binary. This change is needed to train the models.
# 
# Assign a `0` for all `retained` users.
# 
# Assign a `1` for all `churned` users.
# 
# Save this variable as `label2` so as not to overwrite the original `label` variable.
# 
# **Note:** There are many ways to do this. Consider using `np.where()` as you did earlier in this notebook.

# In[19]:


# Create binary `label2` column
df['label2'] = np.where(df['label'] == 'churned', 1, 0)


# ### **Task 6. Feature selection**
# 
# Tree-based models can handle multicollinearity, so the only feature that can be cut is `ID`, since it doesn't contain any information relevant to churn.
# 
# Note, however, that `device` won't be used simply because it's a copy of `device2`.
# 
# Drop `ID` from the `df` dataframe.

# In[20]:


# Drop `ID` column
df = df.drop('ID', axis=1)


# ### **Task 7. Evaluation metric**
# 
# Before modeling, you must decide on an evaluation metric. This will depend on the class balance of the target variable and the use case of the model.
# 
# First, examine the class balance of your target variable.

# In[21]:


# Get class balance of 'label' col
df['label'].value_counts(normalize=True)


# Approximately 18% of the users in this dataset churned. This is an unbalanced dataset, but not extremely so. It can be modeled without any class rebalancing.
# 
# Now, consider which evaluation metric is best. Remember, accuracy might not be the best gauge of performance because a model can have high accuracy on an imbalanced dataset and still fail to predict the minority class.
# 
# It was already determined that the risks involved in making a false positive prediction are minimal. No one stands to get hurt, lose money, or suffer any other significant consequence if they are predicted to churn. Therefore, select the model based on the recall score.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 8. Modeling workflow and model selection process**
# 
# The final modeling dataset contains 14,299 samples. This is towards the lower end of what might be considered sufficient to conduct a robust model selection process, but still doable.
# 
# 1. Split the data into train/validation/test sets (60/20/20)
# 
# Note that, when deciding the split ratio and whether or not to use a validation set to select a champion model, consider both how many samples will be in each data partition, and how many examples of the minority class each would therefore contain. In this case, a 60/20/20 split would result in \~2,860 samples in the validation set and the same number in the test set, of which \~18%&mdash;or 515 samples&mdash;would represent users who churn.
# 2. Fit models and tune hyperparameters on the training set
# 3. Perform final model selection on the validation set
# 4. Assess the champion model's performance on the test set
# 
# ![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)

# ### **Task 9. Split the data**
# 
# Now you're ready to model. The only remaining step is to split the data into features/target variable and training/validation/test sets.
# 
# 1. Define a variable `X` that isolates the features. Remember not to use `device`.
# 
# 2. Define a variable `y` that isolates the target variable (`label2`).
# 
# 3. Split the data 80/20 into an interim training set and a test set. Don't forget to stratify the splits, and set the random state to 42.
# 
# 4. Split the interim training set 75/25 into a training set and a validation set, yielding a final ratio of 60/20/20 for training/validation/test sets. Again, don't forget to stratify the splits and set the random state.

# In[23]:


# 1. Isolate X variables
X = df.drop(['label', 'label2', 'device'], axis=1)

# 2. Isolate y variable
y = df['label2']

# 3. Split into train and test sets
X_interim, X_test, y_interim, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Split into train and validate sets
X_train, X_val, y_train, y_val = train_test_split(X_interim, y_interim, test_size=0.25, random_state=42, stratify=y_interim)


# Verify the number of samples in the partitioned data.

# In[24]:


print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# This aligns with expectations.

# ### **Task 10. Modeling**

# #### **Random forest**
# 
# Begin with using `GridSearchCV` to tune a random forest model.
# 
# 1. Instantiate the random forest classifier `rf` and set the random state.
# 
# 2. Create a dictionary `cv_params` of any of the following hyperparameters and their corresponding values to tune. The more you tune, the better your model will fit the data, but the longer it will take.
#  - `max_depth`
#  - `max_features`
#  - `max_samples`
#  - `min_samples_leaf`
#  - `min_samples_split`
#  - `n_estimators`
# 
# 3. Define a list `scoring` of scoring metrics for GridSearch to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `rf_cv`. Pass to it as arguments:
#  - estimator=`rf`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit=_`)
# 
#  `refit` should be set to `'recall'`.<font/>
# 

# **Note:** If your model fitting takes too long, try reducing the number of options to search over in the grid search.

# In[28]:


# 1. Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {
    'max_depth': [5, 10, 15],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 200, 300]
}

# 3. Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
rf_cv = GridSearchCV(estimator=rf, param_grid=cv_params, scoring=scoring, cv=5, refit='recall')


# Now fit the model to the training data.

# In[29]:


rf_cv.fit(X_train, y_train)


# Examine the best average score across all the validation folds.

# In[27]:


# Examine best score
rf_cv.best_score_


# Examine the best combination of hyperparameters.

# In[31]:


# Examine best hyperparameter combo
rf_cv.best_params_


# Use the `make_results()` function to output all of the scores of your model. Note that the function accepts three arguments.
# 
# This function is provided for you, but if you'd like to challenge yourself, try writing your own function!

# <details>
#   <summary><h5>HINT</h5></summary>
# 
# To learn more about how this function accesses the cross-validation results, refer to the [`GridSearchCV` scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV) for the `cv_results_` attribute.
# 
# </details>

# In[32]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy',
                   }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          },
                         )

    return table


# Pass the `GridSearch` object to the `make_results()` function.

# In[33]:


# Call 'make_results()' on the GridSearch object
rf_results = make_results('Random Forest', rf_cv, 'recall')
rf_results


# Asside from the accuracy, the scores aren't that good. However, recall that when you built the logistic regression model in the last course the recall was \~0.09, which means that this model has 33% better recall and about the same accuracy, and it was trained on less data.
# 
# If you want, feel free to try retuning your hyperparameters to try to get a better score. You might be able to marginally improve the model.

# #### **XGBoost**
# 
#  Try to improve your scores using an XGBoost model.
# 
# 1. Instantiate the XGBoost classifier `xgb` and set `objective='binary:logistic'`. Also set the random state.
# 
# 2. Create a dictionary `cv_params` of the following hyperparameters and their corresponding values to tune:
#  - `max_depth`
#  - `min_child_weight`
#  - `learning_rate`
#  - `n_estimators`
# 
# 3. Define a list `scoring` of scoring metrics for grid search to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `xgb_cv`. Pass to it as arguments:
#  - estimator=`xgb`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit='recall'`)

# In[35]:


# 1. Instantiate the XGBoost classifier
from xgboost import XGBClassifier

# 1. Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

# 3. Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(estimator=xgb, param_grid=cv_params, scoring=scoring, cv=5, refit='recall')


# Now fit the model to the `X_train` and `y_train` data.
# 
# Note this cell might take several minutes to run.

# In[36]:


xgb_cv.fit(X_train, y_train)


# Get the best score from this model.

# In[37]:


# Examine best score
xgb_cv.best_score_


# And the best parameters.

# In[38]:


# Examine best parameters
xgb_cv.best_params_


# Use the `make_results()` function to output all of the scores of your model. Note that the function accepts three arguments.

# In[44]:


# Call 'make_results()' on the GridSearch object
xgb_results = make_results('XGBoost', xgb_cv, 'recall')
results_table = pd.concat([rf_results, xgb_results])
results_table


# This model fit the data even better than the random forest model. The recall score is nearly double the recall score from the logistic regression model from the previous course, and it's almost 50% better than the random forest model's recall score, while maintaining a similar accuracy and precision score.

# ### **Task 11. Model selection**
# 
# Now, use the best random forest model and the best XGBoost model to predict on the validation data. Whichever performs better will be selected as the champion model.

# #### **Random forest**

# In[45]:


# Use random forest model to predict on validation data
rf_val_preds = rf_cv.best_estimator_.predict(X_val)


# Use the `get_test_scores()` function to generate a table of scores from the predictions on the validation data.

# In[46]:


def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                          })

    return table


# In[47]:


# Get validation scores for RF model
rf_val_scores = get_test_scores('Random Forest Validation', rf_val_preds, y_val)

# Append to the results table
results_table = pd.concat([results_table, rf_val_scores])
results_table


# Notice that the scores went down from the training scores across all metrics, but only by very little. This means that the model did not overfit the training data.

# #### **XGBoost**
# 
# Now, do the same thing to get the performance scores of the XGBoost model on the validation data.

# In[48]:


# Use XGBoost model to predict on validation data
xgb_val_preds = xgb_cv.best_estimator_.predict(X_val)

# Get validation scores for XGBoost model
xgb_val_scores = get_test_scores('XGBoost Validation', xgb_val_preds, y_val)

# Append to the results table
results_table = pd.concat([results_table, xgb_val_scores])
results_table


# Just like with the random forest model, the XGBoost model's validation scores were lower, but only very slightly. It is still the clear champion.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 12. Use champion model to predict on test data**
# 
# Now, use the champion model to predict on the test dataset. This is to give a final indication of how you should expect the model to perform on new future data, should you decide to use the model.

# In[49]:


# Use XGBoost model to predict on test data
xgb_test_preds = xgb_cv.best_estimator_.predict(X_test)

# Get test scores for XGBoost model
xgb_test_scores = get_test_scores('XGBoost Test', xgb_test_preds, y_test)

# Append to the results table
results_table = pd.concat([results_table, xgb_test_scores])
results_table


# The recall was exactly the same as it was on the validation data, but the precision declined notably, which caused all of the other scores to drop slightly. Nonetheless, this is stil within the acceptable range for performance discrepancy between validation and test scores.

# ### **Task 13. Confusion matrix**
# 
# Plot a confusion matrix of the champion model's predictions on the test data.

# In[52]:


# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, xgb_test_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# The model predicted three times as many false negatives than it did false positives, and it correctly identified only 16.6% of the users who actually churned.

# ### **Task 14. Feature importance**
# 
# Use the `plot_importance` function to inspect the most important features of your final model.

# In[51]:


# Plot feature importance
plot_importance(xgb_cv.best_estimator_, max_num_features=10)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()


# The XGBoost model made more use of many of the features than did the logistic regression model from the previous course, which weighted a single feature (`activity_days`) very heavily in its final prediction.
# 
# If anything, this underscores the importance of feature engineering. Notice that engineered features accounted for six of the top 10 features (and three of the top five). Feature engineering is often one of the best and easiest ways to boost model performance.
# 
# Also, note that the important features in one model might not be the same as the important features in another model. That's why you shouldn't discount features as unimportant without thoroughly examining them and understanding their relationship with the dependent variable, if possible. These discrepancies between features selected by models are typically caused by complex feature interactions.
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 
# Even if you cannot use the model to make strong predictions, was the work done in vain? What insights can you report back to stakeholders?

# ### **Task 15. Conclusion**
# 
# Now that you've built and tested your machine learning models, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. Would you recommend using this model for churn prediction? Why or why not?
# 
# 2. What tradeoff was made by splitting the data into training, validation, and test sets as opposed to just training and test sets?
# 
# 3. What is the benefit of using a logistic regression model over an ensemble of tree-based models (like random forest or XGBoost) for classification tasks?
# 
# 4. What is the benefit of using an ensemble of tree-based models like random forest or XGBoost over a logistic regression model for classification tasks?
# 
# 5. What could you do to improve this model?
# 
# 6. What additional features would you like to have to help improve the model?

# Task 15. Conclusion
# 
# 1.Would you recommend using this model for churn prediction? Why or why not?
# Yes, I would recommend using the XGBoost model for churn prediction at Waze. With a recall of 22.3% on the test set, it successfully identifies more than 1 in 5 users who will churn - more than twice the performance of the Random Forest model (10.7%) and significantly better than the previous logistic regression model (~9%). While there's still room for improvement, this level of predictive power provides actionable insights for targeted retention efforts. With a precision of 44%, when the model predicts a user will churn, it's correct nearly half the time, making it efficient for directing resources toward at-risk users.
# 
# 2.What tradeoff was made by splitting the data into training, validation, and test sets as opposed to just training and test sets?
# By splitting the data into training (60%), validation (20%), and test (20%) sets instead of just training and test sets, we sacrificed some training data volume that could have potentially improved model performance. However, this three-way split provided crucial benefits: it allowed us to evaluate multiple models independently, tune hyperparameters without contaminating our final evaluation data, and get a more reliable estimate of real-world performance. The validation set served as an important intermediate step to select the best model (XGBoost over Random Forest) before confirming its performance on completely unseen test data, which showed the model's generalizability with recall actually improving from 19.9% to 22.3%.
# 
# 3.What is the benefit of using a logistic regression model over an ensemble of tree-based models for classification tasks?
# Logistic regression offers several advantages over tree-based models: (1) Superior interpretability - coefficients directly indicate each feature's impact direction and magnitude; (2) Computational efficiency - faster training times and lower resource requirements; (3) Better performance with limited data; (4) Simplicity in implementation and deployment; (5) Well-calibrated probability estimates; and (6) Less prone to overfitting with appropriate regularization. For business contexts where explaining predictions is as important as making them, or where computational resources are limited, logistic regression remains valuable despite potentially lower predictive performance.
# 
# 4.What is the benefit of using an ensemble of tree-based models like random forest or XGBoost over a logistic regression model for classification tasks?
# Tree-based models provided significant benefits in our churn prediction task: (1) Superior performance - XGBoost achieved more than double the recall of our previous logistic regression model; (2) Automatic handling of non-linear relationships and complex feature interactions without manual transformation; (3) Built-in feature selection and importance ranking; (4) Robustness to outliers and missing values; and (5) No need for feature scaling. Our results confirmed this advantage, with XGBoost identifying sophisticated patterns in user behavior that logistic regression missed, particularly in how features like km_per_hour and navigation to favorites relate to churn risk.
# 
# 5.What could you do to improve this model?
# To improve the model further, we could: (1) Address class imbalance through techniques like SMOTE, class weighting, or adjusted thresholds; (2) Conduct more extensive hyperparameter tuning, especially focusing on parameters that might improve recall; (3) Create additional engineered features based on the most important predictors we've identified (e.g., more detailed speed metrics or favorite usage patterns); (4) Implement model stacking or ensemble methods to combine strengths of multiple models; (5) Collect larger training datasets; (6) Experiment with more advanced algorithms like neural networks; and (7) Implement time-based features to capture seasonal patterns or trends in user behavior.
# 
# 6.What additional features would you like to have to help improve the model?
# Based on our feature importance results, additional data would be valuable: (1) User satisfaction metrics and feedback scores; (2) Detailed navigation efficiency metrics (time saved compared to typical routes); (3) Usage patterns of specific app features beyond favorites; (4) Technical issues or app crashes experienced; (5) Time of day and day of week usage patterns; (6) User demographics; (7) Device performance metrics; (8) Competitive app usage if available; (9) Whether users drive professionally vs. personally; (10) Traffic conditions encountered during drives; and (11) User interaction with notifications. These additional features would help better identify patterns associated with churn risk, particularly around the key areas highlighted by our feature importance analysis: driving efficiency, app engagement, and navigation patterns.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
