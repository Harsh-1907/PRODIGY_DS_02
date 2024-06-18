import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
directory = 'C:/Users/DELL/Desktop/ProdigyInfotech_DataScience/2/titanic'
files = os.listdir(directory)
print(files)
train_filepath = 'C:/Users/DELL/Desktop/ProdigyInfotech_DataScience/2/titanic/train.csv'
test_filepath = 'C:/Users/DELL/Desktop/ProdigyInfotech_DataScience/2/titanic/test.csv'
gender_submission_filepath = 'C:/Users/DELL/Desktop/ProdigyInfotech_DataScience/2/titanic/gender_submission.csv'
train_df = pd.read_csv(train_filepath)
test_df = pd.read_csv(test_filepath)
gender_submission_df = pd.read_csv(gender_submission_filepath)
missing_values_train = train_df.isnull().sum()
print("Missing values in training dataset:\n", missing_values_train)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df['Has_Cabin'] = train_df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
train_df.drop(columns=['Cabin'], inplace=True)
missing_values_train_after = train_df.isnull().sum()
print("\nMissing values in training dataset after handling:\n", missing_values_train_after)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
rare_titles = train_df['Title'].value_counts()[train_df['Title'].value_counts() < 10].index.tolist()
train_df['Title'] = train_df['Title'].apply(lambda x: 'Rare' if x in rare_titles else x)
test_df['Title'] = test_df['Title'].apply(lambda x: 'Rare' if x in rare_titles else x)
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Pclass')
plt.show()
sns.histplot(data=train_df, x='Age', hue='Survived', multiple='stack', bins=20)
plt.title('Survival by Age')
plt.show()
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Survival Rate by Embarked')
plt.show()
numeric_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train_df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
survived_fare = train_df[train_df['Survived'] == 1]['Fare']
not_survived_fare = train_df[train_df['Survived'] == 0]['Fare']
t_stat, p_value = ttest_ind(survived_fare, not_survived_fare, equal_var=False)
print(f"T-test results for Fare by Survived: t-statistic = {t_stat}, p-value = {p_value}")