import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Analysis
from scipy.stats import chi2_contingency
import ppscore

# Feature Processing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb


# Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

# Model Saving
import pickle

# Other Packages
import random
import warnings

warnings.filterwarnings("ignore")


# ## Data Loading

# ### Loading the Train and Test Datasets

# #### Train Dataset

# In[3]:


# Load The Train Dataset
train_df = pd.read_csv("data\Paitients_Files_Train (1).csv")
train_df.head()


# #### Test Dataset

# In[4]:


# Load The Test Dataset
test_df = pd.read_csv("data\Paitients_Files_Test (1).csv")
test_df.head()



train_df.info()



test_df.info()




# In[7]:


# The shape of the train dataset
train_df.shape


# In[8]:


# The shape of the train dataset
test_df.shape


# ### iii. Summary Statistics Datasets

# In[9]:


# Summary Statistics of The Train Dataset
train_df.describe().round(3)


# In[10]:


# Summary Statistics of The Test
test_df.describe().round(3)




# In[11]:


# Check for missing values in the datasets
datasets = {'train': train_df, 'test': test_df}

def show_missing_values(datasets):
    for name, data in datasets.items():
        print(f"Missing values in the {name.capitalize()} dataset:")
        print(data.isnull().sum())
        print('===' * 18)
        print()

show_missing_values(datasets)


# Both datasets do not have any missing values

# ### v. Checking for Duplicates in The Datasets

# In[12]:


# Check for duplicates in the Train dataset
train_duplicates = train_df[train_df.duplicated()]

# Check for duplicates in the Test dataset
test_duplicates = test_df[test_df.duplicated()]

# Display the duplicate rows in the Train dataset, if any
if not train_duplicates.empty:
    print("Duplicate Rows in Train Dataset:")
    print(train_duplicates)
else:
    print("No Duplicate Rows in Train Dataset")

# Display the duplicate rows in the Test dataset, if any
if not test_duplicates.empty:
    print("\nDuplicate Rows in Test Dataset:")
    print(test_duplicates)
else:
    print("No Duplicate Rows in Test Dataset")


# # Univariate Analysis

# ## i. Univariate Analysis for 'PRG' (Number of Pregnancies)

# In[13]:


# Extract the 'PRG' column
prg_values = train_df['PRG']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(prg_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of Number of Pregnancies (PRG)')
plt.xlabel('PRG (Number of Pregnancies)')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for Number of Pregnancies (PRG):')
print(prg_values.describe())


# - The average number of pregnancies is approximately 3.83, suggesting that, on average, patients have had several pregnancies.
# - The range of values varies from 0 (no pregnancies) to a maximum of 17 pregnancies.
# - The majority of patients fall within the range of 1 to 6 pregnancies.

# ## ii. Univariate Analysis for 'PL' (Plasma Glucose Concentration)

# In[14]:


# Extract the 'PL' column
pl_values = train_df['PL']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(pl_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of Plasma Glucose Concentration (PL)')
plt.xlabel('PL (Plasma Glucose Concentration)')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for Plasma Glucose Concentration (PL):')
print(pl_values.describe())


# - The mean plasma glucose concentration is around 120.15 mg/dL.
# - The values range from a minimum of 0 mg/dL (which seems unusual) to a maximum of 198 mg/dL.
# - The standard deviation of 32.68 indicates some variability in glucose levels among patients.

# ## iii. Univariate Analysis for 'PR' (Diastolic Blood Pressure)

# In[15]:


# Extract the 'PR' column
pr_values = train_df['PR']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(pr_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of Diastolic Blood Pressure (PR)')
plt.xlabel('PR (Diastolic Blood Pressure)')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for Diastolic Blood Pressure (PR):')
print(pr_values.describe())


# - The mean diastolic blood pressure is approximately 68.73 mm Hg.
# - The values range from a minimum of 0 mm Hg (which seems unusual) to a maximum of 122 mm Hg.
# - Most patients have diastolic blood pressure levels within the range of 64 to 80 mm Hg.

# ## iv. Univariate Analysis for 'SK' (Triceps Skinfold Thickness)

# In[16]:


# Extract the 'SK' column
sk_values = train_df['SK']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(sk_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of Triceps Skinfold Thickness (SK)')
plt.xlabel('SK (Triceps Skinfold Thickness)')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for Triceps Skinfold Thickness (SK):')
print(sk_values.describe())


# - The mean triceps skinfold thickness is around 20.56 mm.
# - There is a notable spread in skinfold thickness among patients.

# ## v. Univariate Analysis for 'TS' (2-Hour Serum Insulin)

# In[17]:


# Extract the 'TS' column
ts_values = train_df['TS']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(ts_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of 2-Hour Serum Insulin (TS)')
plt.xlabel('TS (2-Hour Serum Insulin)')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for 2-Hour Serum Insulin (TS):')
print(ts_values.describe())


# - The mean 2-hour serum insulin level is approximately 79.46 μU/ml.
# - The values have a wide range, with a minimum of 0 μU/ml and a maximum of 846 μU/ml.
# - The standard deviation of 116.58 suggests significant variability in insulin levels.

# ## vi. Univariate Analysis for 'M11' (Body Mass Index - BMI)

# In[18]:


# Extract the 'M11' column
m11_values = train_df['M11']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(m11_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of Body Mass Index (BMI)')
plt.xlabel('BMI (Body Mass Index)')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for Body Mass Index (BMI):')
print(m11_values.describe())


# - The mean BMI is approximately 31.92, indicating that, on average, patients have a BMI in the overweight range.
# - BMI values vary widely, with a minimum of 0.078 (unusually low) and a maximum of 67.1.
# - The standard deviation of 8.01 suggests substantial variability in BMI among patients.

# ## vii. BD2 (Diabetes Pedigree Function)

# In[19]:


# Extract the 'BD2' column
bd2_values = train_df['BD2']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(bd2_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of Diabetes Pedigree Function (BD2)')
plt.xlabel('BD2 (Diabetes Pedigree Function)')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for Diabetes Pedigree Function (BD2):')
print(bd2_values.describe())


# - The mean diabetes pedigree function value is 0.481, which reflects the diabetes history in family members.
# - Values range from a minimum of 0.078 to a maximum of 2.42.
# - The spread in diabetes pedigree function values indicates varying family histories of diabetes.

# ## viii. Univariate Analysis for 'Age'

# In[20]:


# Extract the 'Age' column
age_values = train_df['Age']

# Plot a histogram
plt.figure(figsize=(10, 5))
sns.histplot(age_values, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print('Summary Statistics for Age:')
print(age_values.describe())


# - The average age of patients is approximately 33.29 years.
# - Ages range from a minimum of 21 years to a maximum of 81 years.
# - Most patients fall within the range of 24 to 40 years.

# ## ix.Univariate Analysis for 'Insurance'

# In[21]:


# Extract the 'Insurance' column
insurance_values = train_df['Insurance']

# Plot the distribution of Insurance
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Insurance', palette='viridis')
plt.title('Distribution of Insurance')
plt.xlabel('Insurance')
plt.ylabel('Count')
plt.show()

# Summary statistics
print('Summary Statistics for Insurance Coverage:')
print(insurance_values.describe())

print ()


# Calculate the counts of patients with and without insurance
insurance_counts = insurance_values.value_counts()

# Total number of patients
total_patients = insurance_counts.sum()  

# Calculate the percentages
percentage_with_insurance = (insurance_counts.loc[1] / total_patients) * 100
percentage_without_insurance = (insurance_counts.loc[0] / total_patients) * 100

# Insights
print("Insights for the Distribution of Insurance:")

# Print the results with percentages in a single statement
print(f"- There are {insurance_counts.loc[1]} patients with insurance (1), which is {percentage_with_insurance:.2f}% of the total, "
      f"and {insurance_counts.loc[0]} patients without insurance (0), which is {percentage_without_insurance:.2f}% of the total.")



# ## x.Univariate Analysis for 'Sepsis'

# In[22]:


# Plot the distribution of Sepsiss
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Sepssis', palette='viridis')
plt.title('Distribution of Sepsiss')
plt.xlabel('Sepsiss')
plt.ylabel('Count')
plt.show()


# Insights
sepsis_counts = train_df['Sepssis'].value_counts()
print("Insights for the Distribution of Sepsiss:")
print(f"- There are {sepsis_counts['Positive']} patients with sepsis (Positive) and {sepsis_counts['Negative']} patients without sepsis (Negative).")
print("- The distribution shows an imbalance, with more patients without sepsis.")


# # Bivariate Analysis

# - The bivariate analysis will focus on investigating the relationship between age and the various health-related variables. 
# 
# 
# - Analyzing the relationship between age and other relevant variables in relation to sepsis is essential because understanding how age interacts with these variables can provide valuable insights into the risk factors, progression, and management of sepsis across different age groups. 
# 
# 
# - Sepsis is a complex medical condition influenced by various factors, and age is a fundamental demographic variable that can significantly impact the likelihood of its occurrence and its outcomes. 
# 
# 
# - Examining age-specific patterns and associations with variables such as blood pressure, plasma glucose concentration, and other health indicators can identify age-related trends, assess the vulnerability of different age groups to sepsis, and tailor medical interventions and preventive strategies accordingly. 
# 
# 
# - This analysis helps us develop a more comprehensive understanding of sepsis and the age-specific dynamics of the disease.
# 
# 
# - To achieve this, we need to group patients into distinct age groups so that we can explore how specific age groups might be associated with different medical complications or characteristics relevant to sepsis.

# In[23]:


# Creating the Age Group Column
# Define age intervals
age_intervals = [20, 40, 60, 80, 90]
age_labels = ['20-39', '40-59', '60-79', '80-90']

# Create labels for age intervals
train_df['Age Group'] = pd.cut(train_df['Age'], bins=age_intervals, labels=age_labels)

# Add a new column 'Age Group' based on age intervals
train_df['Age Group'] = pd.cut(train_df['Age'], bins=age_intervals, labels=age_labels)
train_df.head()



# In[24]:


# Calculate the average number of pregnancies ('PRG') per age group and round to the nearest whole number
average_pregnancies_by_age = train_df.groupby('Age Group')['PRG'].mean().round().reset_index()

# Rename the columns for clarity
average_pregnancies_by_age.columns = ['Age Group', 'Average Pregnancies']

# Display the DataFrame
average_pregnancies_by_age


# In[25]:


# Create a bar plot to visualize the average number of pregnancies ('PRG') by age group
plt.figure(figsize=(12, 6))
sns.barplot(data=train_df, x='Age Group', y='PRG', palette='viridis', ci=None)
plt.title('Average Number of Pregnancies by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Number of Pregnancies (PRG)')
plt.xticks(rotation=0)
plt.show()




# In[26]:


# Calculate the mean 'PL' for each age group
age_vs_pl_mean = train_df.groupby('Age Group')['PL'].mean().reset_index()

# Rename the columns for clarity
age_vs_pl_mean.columns = ['Age Group', 'Mean PL']

# Display the DataFrame
age_vs_pl_mean


# In[27]:


# Calculate the mean 'PL' for each age group
age_vs_pl_mean = train_df.groupby('Age Group')['PL'].mean().reset_index()
age_vs_pl_mean.columns = ['Age Group', 'Mean PL']

# Create a Matplotlib figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create the line plot
plt.plot(age_vs_pl_mean['Age Group'], age_vs_pl_mean['Mean PL'], marker='o', markersize=8, color='black', label='Mean PL', linestyle='-')

# Customize plot aesthetics
plt.title('Mean Plasma Glucose Concentration (PL) by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Mean Plasma Glucose Concentration (PL)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()




# In[28]:


# Calculate the mean 'PR' for each age group
age_vs_pr_mean = train_df.groupby('Age Group')['PR'].mean().reset_index()

# Rename the columns for clarity
age_vs_pr_mean.columns = ['Age Group', 'Mean PR']

# Display the DataFrame
age_vs_pr_mean


# In[29]:


# Calculate the mean 'PR' for each age group
age_vs_pr_mean = train_df.groupby('Age Group')['PR'].mean().reset_index()
age_vs_pr_mean.columns = ['Age Group', 'Mean PR']

# Set the Viridis color palette
viridis_colors = plt.cm.viridis(age_vs_pr_mean.index / len(age_vs_pr_mean))

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(age_vs_pr_mean['Age Group'], age_vs_pr_mean['Mean PR'], color=viridis_colors)
plt.title('Mean Diastolic Blood Pressure (PR) by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Diastolic Blood Pressure (PR)')

# Show the plot
plt.tight_layout()
plt.show()




# In[30]:


# Calculate the mean 'PR' for each age group
age_vs_bmi_mean = train_df.groupby('Age Group')['M11'].mean().reset_index()

# Rename the columns for clarity
age_vs_bmi_mean.columns = ['Age Group', 'Mean M11']

# Display the DataFrame
age_vs_bmi_mean


# In[31]:


# Calculate the mean 'M11' for each age group
age_vs_bmi_mean = train_df.groupby('Age Group')['M11'].mean().reset_index()
age_vs_bmi_mean.columns = ['Age Group', 'Mean M11']

# Set the Viridis color palette
viridis_colors = plt.cm.viridis(age_vs_bmi_mean.index / len(age_vs_bmi_mean))

# Create the bar plot with the Viridis palette
plt.figure(figsize=(10, 6))
plt.bar(age_vs_bmi_mean['Age Group'], age_vs_bmi_mean['Mean M11'], color=viridis_colors)
plt.title('Mean BMI by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean BMI')

# Show the plot
plt.tight_layout()
plt.show()


# - Individuals in the 40-59 age group exhibit the highest mean BMI (33.31), indicating a higher prevalence of overweight or obesity in this age range.
# 
# 
# - In contrast, those aged 80-90 have the lowest mean BMI (25.90), suggesting a trend toward lower body weight in this older age group.
# 
# - The 20-39 age group shows a moderately high mean BMI (31.69), indicating a considerable proportion of individuals with higher body weight.
# 
# 
# - Interestingly, the 60-79 age group falls in between, with a mean BMI of 28.79.
# 
# 
# The relationship between age and BMI highlights that BMI tends to increase from younger to middle-aged adults, peaking in the 40-59 age group, and then gradually decreasing in the older age categories. This pattern may reflect lifestyle changes, metabolic shifts, or medical conditions associated with aging.
# 
# With regards to Sepsis, middle-aged individuals may have specific risks related to their BMI, while older age groups may have different risk factors to consider in sepsis management.

# ## v. Bivariate Analysis between 'Age' and 'BD2' (Age vs. Diabetes Pedigree Function)

# Understanding how age interacts with BD2, a measure associated with the genetic predisposition to diabetes, is valuable for gaining insights into the risk factors and potential connections between age and diabetes-related health conditions, which can influence the likelihood and severity of sepsis. Analyzing this relationship allows us to assess whether BD2 levels vary significantly across different age groups, providing a better understanding of how age-related factors might contribute to the susceptibility or management of sepsis in various age cohorts. By investigating age-specific patterns in BD2, we aim to uncover potential correlations that could inform sepsis risk assessments and interventions tailored to specific age demographics.

# In[32]:


# Calculate the mean 'BD2' for each age group
age_vs_bd2_mean = train_df.groupby('Age Group')['BD2'].mean().reset_index()

# Rename the columns for clarity
age_vs_bd2_mean.columns = ['Age Group', 'Diabetes Pedigree Function']


age_vs_bd2_mean


# In[33]:


# Calculate the mean 'BD2' for each age group
age_vs_bd2_mean = train_df.groupby('Age Group')['BD2'].mean().reset_index()

# Set the Viridis color palette
viridis_colors = plt.cm.viridis(age_vs_bd2_mean.index / len(age_vs_bd2_mean))

# Create the bar plot with the Viridis palette
plt.figure(figsize=(10, 6))
plt.bar(age_vs_bd2_mean['Age Group'], age_vs_bd2_mean['BD2'], color=viridis_colors)
plt.title('Diabetes Pedigree Function (BD2) by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Diabetes Pedigree Function')

# Show the plot
plt.tight_layout()
plt.show()




# In[34]:


# Separate positive and negative cases into two DataFrames
positive_df = train_df[train_df['Sepssis'] == 'Positive']
negative_df = train_df[train_df['Sepssis'] == 'Negative']

# Create DataFrames showing the counts for each age group
sepsis_positive_counts = positive_df['Age Group'].value_counts().reset_index()
sepsis_negative_counts = negative_df['Age Group'].value_counts().reset_index()

# Display the DataFrames
print("Sepsis Positive Counts:")
sepsis_positive_counts


# In[35]:


# Separate positive and negative cases into two DataFrames
positive_df = train_df[train_df['Sepssis'] == 'Positive']
negative_df = train_df[train_df['Sepssis'] == 'Negative']

# Create DataFrames showing the counts for each age group
sepsis_positive_counts = positive_df['Age Group'].value_counts().reset_index()
sepsis_positive_counts.columns = ['age group', 'count']

sepsis_negative_counts = negative_df['Age Group'].value_counts().reset_index()
sepsis_negative_counts.columns = ['age group', 'count']

# Display the DataFrames
print("Sepsis Positive Counts:")
sepsis_positive_counts


# In[36]:


print("\nSepsis Negative Counts:")
sepsis_negative_counts


# In[37]:


# Create a line plot
plt.figure(figsize=(12, 6))
plt.plot(sepsis_positive_counts['age group'], sepsis_positive_counts['count'], marker='o', color='red', label='Positive Sepsis Count')
plt.plot(sepsis_negative_counts['age group'], sepsis_negative_counts['count'], marker='o', color='green', label='Negative Sepsis Count')
plt.title('Trend of Sepsis by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()


# In[38]:


# Group the data by 'Age Group' and count the number of participants in each group
age_group_counts = train_df['Age Group'].value_counts().reset_index()
age_group_counts.columns = ['Age Group', 'Participant Count']

# Sort the DataFrame by 'Age Group'
age_group_counts = age_group_counts.sort_values(by='Age Group')

# Display the DataFrame.
age_group_counts



# In[39]:


# Create a contingency table of Age vs. Sepsis
contingency_table = pd.crosstab(train_df['Age Group'], train_df['Sepssis'])

# Perform the Chi-Square test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Define the significance level (alpha)
alpha = 0.05

# Check if the p-value is less than alpha
if p < alpha:
    print("Reject the null hypothesis")
    print("There is a significant association between age and sepsis.")
else:
    print("Fail to reject the null hypothesis")
    print("There is no significant association between age and sepsis.")

# Print the Chi-Square statistic and p-value
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")


# The Chi-Square test results indicate that we should reject the null hypothesis, as the p-value (approximately 2.46e-07) is significantly smaller than the chosen significance level (alpha) of 0.05. This means that there is a statistically significant association between a patient's age (in different age groups) and the likelihood of sepsis.

# ## Answering Key Analytical Questions

# To address our key analytical objectives, we will focus on examining the interrelationships between various variables. We will utilize the Pearson Correlation coefficient, a foundational tool for quantifying the strength and direction of linear associations among these variables. Additionally, a correlation matrix will be constructed to provide a structured representation of these correlations. This matrix will offer a comprehensive view of how different medical parameters are interconnected. These critical insights, obtained from the Pearson Correlation coefficient and the correlation matrix, will empower us to discern the factors that will significantly influence the risk of sepsis.

# ### i. Are there any correlations or patterns between the numerical features (e.g., PRG, PL, PR, SK, TS, M11, BD2, Age) and the presence of sepsis (Positive/Negative)?
# 

# In[40]:


# Calculate the correlation matrix
corr_matrix = train_df[['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Age and Number of Pregnancies (PRG): There is a moderately positive correlation (0.53) between the number of pregnancies (PRG) and age. This suggests that, on average, older individuals tend to have more pregnancies.
# 
# Plasma Glucose (PL) and 2-Hour Serum Insulin (TS): Plasma glucose (PL) and 2-hour serum insulin (TS) exhibit a moderate positive correlation (0.34). This correlation indicates that higher plasma glucose levels are associated with higher 2-hour serum insulin levels.
# 
# Plasma Glucose (PL) and Diastolic Blood Pressure (PR): There is a positive correlation (0.14) between plasma glucose (PL) and diastolic blood pressure (PR). This suggests that higher plasma glucose levels may be associated with higher diastolic blood pressure.
# 
# Skinfold Thickness (SK) and 2-Hour Serum Insulin (TS): Skinfold thickness (SK) and 2-hour serum insulin (TS) have a relatively strong positive correlation (0.43). This implies that individuals with higher skinfold thickness may tend to have higher 2-hour serum insulin levels.
# 
# Age and Skinfold Thickness (SK): There is a negative correlation (-0.12) between age and skinfold thickness (SK). This suggests that, on average, older individuals may have lower skinfold thickness.
# 
# Pregnancies (PRG) and Diastolic Blood Pressure (PR): The correlation between the number of pregnancies (PRG) and diastolic blood pressure (PR) is relatively low (0.12), indicating a weak positive relationship.
# 
# Age and Diabetes Pedigree Function (BD2): Age and the diabetes pedigree function (BD2) have a weak positive correlation (0.05), suggesting that age and the diabetes pedigree function are somewhat related, but the relationship is not strong.

# ### ii. How does the distribution of key numerical variables (e.g., PR, SK, TS, M11) differ between patients with and without sepsis?

# In[41]:


# Create subplots for each numerical variable
plt.figure(figsize=(12, 8))

# Define the numerical variables of interest
numerical_variables = ['PR', 'SK', 'TS', 'M11']

# Define the color palette with green and blue from the 'viridis' palette
colors = sns.color_palette('viridis', n_colors=2)

# Loop through the numerical variables and create subplots
for i, variable in enumerate(numerical_variables):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Sepssis', y=variable, data=train_df, hue='Sepssis', palette=colors)
    plt.title(f'Distribution of {variable} by Sepsis')
    plt.xlabel('Sepsis Status')
    plt.ylabel(variable)
    
plt.tight_layout()
plt.show()


# The boxplots reveal that patients with sepsis generally have higher  PR, SK, TS and M11 compared to patients without Sepsis. These differences suggest that PR, SK, TS, and BMI might be relevant factors in assessing the risk of sepsis.

# ### iii. Is there a relationship between the number of pregnancies (PRG) and plasma glucose concentration (PL)? Does this relationship vary with the presence of sepsis?
# 

# In[42]:


# Select the columns of interest
scatter_data = train_df[['PRG', 'PL']]

# Create a scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PRG', y='PL', data=scatter_data)
plt.title('Scatterplot of PRG vs. PL')
plt.xlabel('PRG')
plt.ylabel('PL')
plt.show()

# Calculate Pearson correlation
s0 = scatter_data['PRG']
s1 = scatter_data['PL']
pearson = s0.corr(s1, method='pearson')

# Print the correlation values
print(f"Pearson Correlation between 'PRG' and 'PL': {pearson}")


# The Pearson correlation indicate a weak positive relationship between 'PRG' and 'PL.' While the Pearson correlation measures linear association and Spearman is non-parametric, they provide consistent results in this case, suggesting that as 'PRG' values increase, 'PL' values tend to increase, but the relationship is not particularly strong. These results are corroborated by the correlation matrix.

# ### iv. Are there any significant differences in diastolic blood pressure (PR) between patients with different triceps skinfold thickness (SK) levels?

# In[43]:


# Select the columns of interest
scatter_data = train_df[['PR', 'SK']]

# Create a scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='SK', y='PR', data=scatter_data)
plt.title('Scatterplot of SK vs. PR')
plt.xlabel('SK')
plt.ylabel('PR')
plt.show()

# Calculate Pearson correlation
s0 = scatter_data['PR']
s1 = scatter_data['SK']
pearson = s0.corr(s1, method='pearson')

# Print the correlation values
print(f"Pearson Correlation between 'PR' and 'SK': {pearson}")


# The analysis indicates a modest positive relationship (Pearson correlation coefficient of approximately 0.198) between diastolic blood pressure (PR) and triceps skinfold thickness (SK). This suggests that as triceps skinfold thickness increases, diastolic blood pressure tends to rise slightly. Such a correlation may have potential implications for cardiovascular health, as higher triceps skinfold thickness often reflects increased subcutaneous fat. However, it's essential to recognize that the correlation is not notably strong, and its clinical significance would require further investigation and the consideration of additional health factors to draw comprehensive conclusions about the potential impact on individuals' health and well-being.

# ### v. Does the body mass index (M11) vary significantly with 2-hour serum insulin (TS) levels?
# 

# In[44]:


# Scatterplot of M11 (BMI) vs. TS
plt.figure(figsize=(8, 5))
sns.scatterplot(data=train_df, x='TS', y='M11')
plt.xlabel('TS (2-Hour Serum Insulin)')
plt.ylabel('M11 (BMI)')
plt.title('Variation of BMI (M11) with TS')
plt.show()

# Calculate Pearson correlation
s0 = train_df['TS']  
s1 = train_df['M11']  
pearson = s0.corr(s1, method='pearson')

# Print the correlation values
print(f"Pearson Correlation between TS and M11: {pearson}")


# The Pearson Correlation between 'TS' (2-Hour Serum Insulin) and 'M11' (BMI) is approximately 0.185. This indicates a weak positive linear correlation between these two variables. The correlation of 0.185 suggests that as 2-Hour Serum Insulin levels ('TS') increase, there is a slight tendency for BMI ('M11') to also increase, but the relationship is not notably strong. It's important to note that correlation does not imply causation, so while there is an association, it doesn't necessarily mean that changes in 2-Hour Serum Insulin directly cause changes in BMI.

# ### vi. Is there a correlation between the diabetes pedigree function (BD2) and age? How does this correlation affect the likelihood of sepsis?

# In[45]:


# Create a scatterplot to visualize the relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(data=train_df, x='BD2', y='Age', hue='Sepssis')
plt.xlabel('BD2 (Diabetes Pedigree Function)')
plt.ylabel('Age')
plt.title('Relationship Between BD2 and Age, Colored by Sepsis')
plt.show()

# Calculate Pearson correlation between BD2 and age
pearson_correlation = train_df['BD2'].corr(train_df['Age'], method='pearson')

# Print the Pearson correlation value
print(f"Pearson Correlation between BD2 and Age: {pearson_correlation}")


# The Pearson Correlation between the diabetes pedigree function (BD2), a measure of genetic diabetes susceptibility, and age is extremely weak at approximately 0.0336. This weak correlation suggests that age has minimal influence on BD2. Consequently, when considering sepsis risk solely in terms of age and its correlation with BD2, it is unlikely that age significantly affects the likelihood of sepsis based on BD2. In this specific context, the focus is on the limited relationship between age and BD2, indicating that age alone may not be a reliable predictor for sepsis risk associated with BD2. Other factors may play a more substantial role in determining sepsis risk.

# ### vii. Are patients with insurance coverage more likely to have certain health characteristics (e.g., higher age, higher BMI) compared to those without insurance coverage?

# In[46]:


# Numeric variables to compare
numeric_variables = ['PRG', 'PL', 'TS', 'M11']

# Create subplots for each numeric variable
plt.figure(figsize=(15, 10))
colors = sns.color_palette('viridis', n_colors=2)

for i, variable in enumerate(numeric_variables):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=train_df, x='Insurance', y=variable, palette=colors)
    plt.xlabel('Insurance Coverage')
    plt.ylabel(variable)
    plt.title(f'{variable} Distribution by Insurance Coverage')

plt.tight_layout()
plt.show()

# Calculate the mean for each numeric variable based on Insurance (0, 1)
mean_numeric_vs_insurance = train_df.groupby('Insurance')[numeric_variables].mean().reset_index()
mean_numeric_vs_insurance.columns = ['Insurance'] + [f'Mean {var}' for var in numeric_variables]

# Display the DataFrame
mean_numeric_vs_insurance





# In[47]:


# Define age intervals
age_intervals = [20, 40, 60, 80, 90]
age_labels = ['20-39', '40-59', '60-79', '80-90']

# Create 'Age Group' column in test_df based on age intervals
test_df['Age Group'] = pd.cut(test_df['Age'], bins=age_intervals, labels=age_labels)

# Now you can proceed with the imputation code
# Define columns to impute
columns_to_impute = ['PL', 'PR', 'SK', 'M11']

# Loop through each column for imputation
for column in columns_to_impute:
    # Identify rows with a value of 0 in the column for both train and test datasets
    zero_rows_train = train_df[train_df[column] == 0]
    zero_rows_test = test_df[test_df[column] == 0]
    
    # Iterate through age groups
    for age_group in age_labels:
        # Calculate the mean of the column within the specific age group for both datasets
        age_group_mean_train = train_df[train_df['Age Group'] == age_group][column].mean()
        age_group_mean_test = test_df[test_df['Age Group'] == age_group][column].mean()
        
        # Impute the mean for rows in the specific age group for both datasets
        train_df.loc[(train_df['Age Group'] == age_group) & (train_df[column] == 0), column] = age_group_mean_train
        test_df.loc[(test_df['Age Group'] == age_group) & (test_df[column] == 0), column] = age_group_mean_test


# In[48]:


train_df.describe()


# In[49]:


test_df.describe()


# The zero values in the four columns have been handled as the minimum values in these columns in the train and test datasets are no longer 0.

# ## ii. Creating New Features For Visualization

# ### a. BMI Categorization

# In[50]:


# BMI Categorization
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

train_df['BMICategory'] = train_df['M11'].apply(categorize_bmi)
test_df['BMICategory'] = test_df['M11'].apply(categorize_bmi)


# The categorization is based on the World Health Organization (WHO) guidelines, which are widely used to classify individuals into different weight categories.
# The categories are defined as follows:
# - Underweight: 
# 
# BMI less than 18.5 - Individuals in this category may be considered underweight, indicating that they have a lower body weight relative to their height.
# 
# - Normal: 
# 
# BMI between 18.5 and 25 - This is considered a healthy weight range, indicating that the individual's weight is proportionate to their height.
# 
# - Overweight: 
# 
# BMI between 25 and 30 - Individuals in this category may be classified as overweight, suggesting that they have an excess of body weight relative to their height.
# 
# - Obese:
# BMI greater than or equal to 30 - This category includes individuals with a high degree of body fat relative to their height.
# 
# The categorization allows for an assessment of an individual's weight status, which can be useful in understanding health risks associated with different weight categories.

# ### b. Glucose Level Categorization

# In[51]:


# Glucose Level Categorization
def categorize_glucose(pl):
    if pl < 100:
        return 'Normal'
    elif 100 <= pl < 126:
        return 'Prediabetic'
    else:
        return 'Diabetic'

train_df['GlucoseCategory'] = train_df['PL'].apply(categorize_glucose)
test_df['GlucoseCategory'] = test_df['PL'].apply(categorize_glucose)


# This categorization is based on the measurement of Plasma Glucose (PL) levels, which is an indicator of blood sugar concentration and is often used in the context of assessing an individual's risk of diabetes or prediabetes.
# The categories are defined as follows:
# 
# - Normal: 
# 
# PL level less than 100 mg/dL - This indicates a normal or healthy blood sugar level.
# 
# - Prediabetic:
# 
# PL level between 100 and 125 mg/dL - This suggests elevated blood sugar levels, which may indicate a risk of developing diabetes in the future.
# 
# - Diabetic: 
# 
# PL level 126 mg/dL or higher - Individuals in this category are typically diagnosed with diabetes.
# 
# Monitoring glucose levels and categorizing them helps in identifying individuals at risk of diabetes.

# ### c. Diastolic Blood Pressure Categorization

# In[52]:


# Diastolic Blood Pressure Category
def categorize_pr(pr):
    if pr < 80:
        return 'Normal'
    elif 80 <= pr < 90:
        return 'Prehypertension'
    elif 90 <= pr < 100:
        return 'Stage1Hypertension'
    else:
        return 'Stage2Hypertension'

train_df['PRCategory'] = train_df['PR'].apply(categorize_pr)
test_df['PRCategory'] = test_df['PR'].apply(categorize_pr)




# In[53]:


# Interaction Feature (Age x Plasma Glucose Concentration (PL))
train_df['Age_PL_Interact'] = train_df['Age'] * train_df['PL']
test_df['Age_PL_Interact'] = test_df['Age'] * test_df['PL']


# The multiplication of Age x Plasma Glucose Concentration (PL) to come up with 'Age_PL_Interact' is a metric that captures the relationship between age and plasma glucose concentration (PL) and how they affect the target variable, which is sepsis risk. 
# 
# 
# This metric is based on the following observations:
# - As people age, their bodies become less efficient at regulating blood sugar levels. This can lead to increased plasma glucose concentrations.
# - Older adults are more likely to have underlying medical conditions that can increase their risk of sepsis, such as diabetes and heart disease.
# 
# This metric is also known as the Age-Plasma Glucose (APG) product. It has been used in a number of studies to predict sepsis risk.
# - A patient with a higher age and plasma glucose concentration (APG) product has a higher risk of sepsis.

# In[54]:


train_df.head()


# In[55]:


test_df.head()


# ### e. Saving The Datasets For PowerBI Visualization

# In[56]:


# Save the updated datasets
train_df.to_csv("data/Visualization_Data_Train.csv", index=False)
test_df.to_csv("data/Visualization_Data_Test.csv", index=False)


# ### f. Dropping The New Features Created For Visualization and Other Unneccessary Columns

# In[57]:


# Drop unnecessary columns
train_df.drop(['ID', 'Age Group', 'Insurance', 'BMICategory', 'GlucoseCategory', 'PRCategory', 'Age_PL_Interact' ], axis=1, inplace=True)
test_df.drop(['ID', 'Age Group', 'Insurance', 'BMICategory', 'GlucoseCategory', 'PRCategory', 'Age_PL_Interact' ], axis=1, inplace=True)


# In[58]:


train_df.head()


# In[59]:


test_df.head()


# ## iii. Encoding Categorical Variables

# In[60]:


# Initialize LabelEncoder for the target variable 'Sepssis'
label_encoder = LabelEncoder()

# Encode the target variable
train_df['Sepssis'] = label_encoder.fit_transform(train_df['Sepssis'])


# In[61]:


# Display the updated train dataset
train_df.head()


# In[62]:


# Display the updated test dataset
test_df.head()


# The Columns have been encoded and are ready for modeling

# # Modeling

# ## Dataset Splitting

# In[63]:


# Define your features (X) and target variable (y)
X = train_df.drop('Sepssis', axis=1)
y = train_df['Sepssis']

# Split the data into training and evaluation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Shape of the Training Set (X_train, y_train):", X_train.shape, y_train.shape)
print("Shape of the Evaluation Set (X_eval, y_eval):", X_eval.shape, y_eval.shape)


# ## Balancing the Training Set

# In[64]:


# Check the class distribution in the training set
print("Class distribution before balancing:")
print(y_train.value_counts())


# In[65]:


# Count the occurrences of each class in the dataset
class_counts = y_train.value_counts()

# Create a bar plot to visualize the class distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.xlabel('Sepssis')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# In[66]:


# Balance the training set using Random Oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

# Check the class distribution after balancing
balanced_class_counts = y_train_balanced.value_counts()
print("\nClass distribution in the balanced training set:")
print(balanced_class_counts)


# In[67]:


# Count the occurrences of each class in the balanced dataset
class_counts = y_train_balanced.value_counts()

# Create a bar plot with the 'viridis' color palette
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.xlabel('Sepssis')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# The train set is now balanced.

# ## Scaling The Training and Evaluation Sets

# In[68]:


# Scale the training and evaluation sets
scaler = MinMaxScaler()

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train_balanced)

# Scale the evaluation data using the same scaler
X_eval_scaled = scaler.transform(X_eval)


# In[69]:


# Convert scaled NumPy arrays back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_eval_scaled_df = pd.DataFrame(X_eval_scaled, columns=X.columns)

# Now, X_train_scaled_df and X_eval_scaled_df are DataFrames containing the scaled features.

# Additionally, convert the balanced target variable back to a DataFrame as well
y_train_balanced_df = pd.DataFrame(y_train_balanced, columns=['Sepssis'])

# Now, y_train_balanced_df is a DataFrame containing the balanced target variable.

# Confirm the shapes of the DataFrames
print("Shape of X_train_scaled_df:", X_train_scaled_df.shape)
print("Shape of X_eval_scaled_df:", X_eval_scaled_df.shape)
print("Shape of y_train_balanced_df:", y_train_balanced_df.shape)
print("Shape of y_eval:", y_eval.shape)


# In[70]:


# View the first few rows of the scaled X_train
X_train_scaled_df.head()


# In[71]:


# View the first few rows of the scaled X_eval
X_eval_scaled_df.head()


# ## Model Training and Evaluation

# ### Model Training

# In[72]:


# Initialize an empty dictionary named 'Results' to store the evaluation results for different models.
Results = {'Model':[], 'Acurracy':[], 'Precision':[], 'Recall':[], 'F1':[]}


# In[73]:


# Converting the dictionary Results into a pandas DataFrame.
Results = pd.DataFrame(Results)
Results.head()


# In[74]:


# Machine Learning Models Initialization
# Logistic Regression
lr = LogisticRegression(max_iter=1000)

# Random Forest
rf = RandomForestClassifier()

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Decision Tree Classifier
dt = DecisionTreeClassifier()

# Gradient Boosting Classifier
gb = GradientBoostingClassifier()

# Gaussian Naive Bayes
nb = GaussianNB()

# Support Vector Machine
svm = SVC()


# In[75]:


# List of model names
models = ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbors', 'Decision Tree', 
          'Gradient Boosting Classifier', 'Gaussian Naive Bayes', 'Support Vector Machine']

# Initialize the Results DataFrame
Results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

# Fit the models and evaluate the performance
for model_name, model in zip(models, [lr, rf, knn, dt, gb, nb, svm]):
    model.fit(X_train_scaled_df, y_train_balanced_df)
    y_pred = model.predict(X_eval_scaled_df)

    print('Model:', model_name)
    print('=' * 50)
    
    print('Confusion Matrix:')
    print(confusion_matrix(y_eval, y_pred))
    
    print('Classification Report:')
    print(classification_report(y_eval, y_pred))

    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)

    print('Accuracy:', round(accuracy, 2))
    print('Precision:', round(precision, 2))
    print('Recall:', round(recall, 2))
    print('F1:', round(f1, 2))

    # Generate the confusion matrix
    cm = confusion_matrix(y_eval, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Display the confusion matrix
    plt.show()

    print("\n")

    # Append the results to the DataFrame with rounding to 2 decimal places
    new_row = {
        'Model': model_name,
        'Accuracy': round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1': round(f1, 2)
    }

    Results = pd.concat([Results, pd.DataFrame([new_row])], ignore_index=True)
    
# Print the results in a leaderboard format based on the F1 score from highest to lowest
print("\nLeaderboard (Ranked by F1 Score)")
Results_sorted = Results.sort_values(by='F1', ascending=False).reset_index(drop=True)

# Rename the "F1" column to "F1-Score" in the Results DataFrame
Results_sorted.rename(columns={'F1': 'F1-Score'}, inplace=True)

# Print The Sorted Results Dataframe
Results_sorted




# In[91]:


# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Fit a Gradient Boosting Classifier
gb.fit(X_train_scaled_df, y_train_balanced_df)

# Get feature importances
feature_importances = gb.feature_importances_

# Calculate the sum of importances for percentage calculation
total_importance = feature_importances.sum()

# Create a DataFrame with correct feature names and percentage importances
feature_importance_df = pd.DataFrame({'Feature': X_train_scaled_df.columns, 'Importance': feature_importances})
feature_importance_df['Percentage'] = ((feature_importance_df['Importance'] / total_importance) * 100).round(2)  # Round to 2 decimal places
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Set the Viridis color palette
sns.set_palette("viridis")

# Visualize feature importances with rounded percentage values using Viridis palette
plt.figure(figsize=(10, 6))
sns.barplot(x='Percentage', y='Feature', data=feature_importance_df, palette="viridis")
plt.title('Feature Importances (Percentage)')
plt.xlabel('Percentage')
plt.ylabel('Feature')
plt.show()

# Print out the exact rounded percentage values
print("Exact Percentage Feature Importances (Rounded to 2 decimal places):")
feature_importance_df


# i. **Plasma Glucose Concentration (PL):**
#    - This feature has the highest importance score, accounting for approximately 41.99% of the predictive power of the model. It suggests that variations in plasma glucose concentration have a significant impact on predicting sepsis.
# 
# ii. **Body Mass Index (M11):**
#    - The second most important feature, with an importance score of around 18.61%. This indicates that BMI plays a substantial role in predicting sepsis, with higher BMI values likely associated with the condition.
# 
# iii. **Diabetes Pedigree Function (BD2):**
#    - While not as influential as PL and M11, this feature still contributes significantly, with approximately 11.32% importance. It implies that the diabetes pedigree function has a notable impact on sepsis predictions.
# 
# iv. **Age:**
#    - Age is an important factor, with roughly 9.43% importance. This suggests that older individuals may be more susceptible to sepsis, and their age is a valuable predictor.
# 
# v. **Number of Pregnancies (PRG):**
#    - Although it has a lower importance score (around 3.72%), the number of pregnancies still contributes to predicting sepsis. It may indicate that pregnancy history can influence sepsis risk.
# 
# vi. **2-Hour Serum Insulin (TS):**
#    - TS has an importance score of about 6.90%, suggesting it plays a minor role compared to other features. Nevertheless, it still contributes to the model's predictive performance.
# 
# vii. **Triceps Skinfold Thickness (SK):**
#    - This feature has a similar importance score to TS, around 4.37%. It may not be as crucial as PL or M11, but it still provides relevant information for sepsis prediction.
# 
# viii. **Diastolic Blood Pressure (PR):**
#    - PR has the lowest importance score, approximately 3.66%. While it is the least influential feature, it still contributes modestly to the model's predictions.
# 
# These insights highlight the relative importance of each feature in predicting sepsis. Features like Plasma Glucose Concentration (PL) and Body Mass Index (M11) are particularly significant, while others, like Diastolic Blood Pressure (PR), have a comparatively lower impact on the model's predictive power.

# #### Hyperparameter Tuning

# In[77]:


# Define hyperparameters to search
param_grid = {
    'n_estimators': [5, 10, 20],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [5, 6, 7],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8]
}

# Create GridSearchCV
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid, 
                           scoring='f1', cv=5, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_scaled_df, y_train_balanced_df)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


# #### Perfomance after Hyperparameter Tuning

# In[93]:


# Instantiate the Gradient Boosting Classifier with the best hyperparameters
tuned_gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.001,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Fit the tuned Gradient Boosting Classifier to the balanced training data
tuned_gb.fit(X_train_scaled, y_train_balanced)

# Evaluate the model on the evaluation set (X_eval_scaled and y_eval)
y_eval_pred = tuned_gb.predict(X_eval_scaled)

# Print the classification report
print("Evaluation Set Classification Report:")
print(classification_report(y_eval, y_eval_pred))

# Print accuracy, precision, recall, and F1-score
accuracy_eval = accuracy_score(y_eval, y_eval_pred)
precision_eval = precision_score(y_eval, y_eval_pred)
recall_eval = recall_score(y_eval, y_eval_pred)
f1_eval = f1_score(y_eval, y_eval_pred)

print("Accuracy:", round(accuracy_eval, 2))
print("Precision:", round(precision_eval, 2))
print("Recall:", round(recall_eval, 2))
print("F1-score:", round(f1_eval, 2))

# Create a confusion matrix heatmap
cm_eval = confusion_matrix(y_eval, y_eval_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_eval, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Evaluation Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("The F1-score After Model Evaluation is:", round(f1_eval, 2))


# The performance of the model improved after hyperparameter tuning. Therefore, we save the tuned model.

# ### Save The Model and Key Components

# In[79]:


# Create a dictionary to store the components
saved_components = {
    'model': tuned_gb,      
    'encoder': label_encoder, 
    'scaler': scaler   
}

# Save all components in a single pickle file
with open('model_and_key_components.pkl', 'wb') as file:
    pickle.dump(saved_components, file)


# #### Test Prediction

# In[80]:


# View the head of the test dataset
test_df.head()


# In[81]:


# Create a copy of the test_df
test_pred = test_df.copy()
test_pred.head()


# In[82]:


# Load the model and component
with open('model_and_key_components.pkl', 'rb') as file:
    loaded_components = pickle.load(file)

# Load the model, encoder and scaler
loaded_model = loaded_components['model']
loaded_encoder = loaded_components['encoder']
loaded_scaler = loaded_components['scaler']

# Encode Categorical Variables
# All Columns are numerical. No need for encoding

# Apply scaling to numerical data
numerical_cols = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']
test_data_scaled = loaded_scaler.transform(test_pred[numerical_cols])

# Convert the scaled numerical data to a DataFrame
test_data_scaled_df = pd.DataFrame(test_data_scaled, columns=numerical_cols)

# Perform Predictions
y_pred = loaded_model.predict(test_data_scaled_df)

# Create a new DataFrame with the 'Sepsis' predictions
test_predictions = test_pred.copy()
test_predictions['Sepsis'] = y_pred

# Define a mapping dictionary
sepsis_mapping = {0: 'Negative', 1: 'Positive'}

# Map the values in the 'Sepsis' column
test_predictions['Sepsis'] = test_predictions['Sepsis'].map(sepsis_mapping)

# Display Predictions (first few rows)
test_predictions.head()


# In[83]:


# Save the updated datasets
test_predictions.to_csv("data/Test_Predictions.csv", index=False)

 

import numpy as np  
import pickle  

# Function to load the model from a file  
def load_model(filename):  
    with open(filename, 'rb') as file:  
        model_data = pickle.load(file)  
    return model_data['model'], model_data['encoder'], model_data['scaler']  

# Function to preprocess input data using the scaler  
def preprocess_input(scaler, input_data):  
    # Convert input_data to a numpy array and reshape for scaling  
    input_array = np.array(input_data).reshape(1, -1)  
    scaled_input = scaler.transform(input_array)  
    return scaled_input  

# Function to check for sepsis  
def check_for_sepsis(classifier, scaler, input_data):  
    # Preprocess input data  
    scaled_input = preprocess_input(scaler, input_data)  

    # Get the prediction  
    prediction = classifier.predict(scaled_input)  
    return "Sepsis detected" if prediction[0] == 1 else "No sepsis detected"  

# Main execution  
if __name__ == "__main__":  
    model_filename = 'model_and_key_components.pkl'  # Adjust this path if necessary  

    # Load the model, encoder, and scaler  
    classifier, encoder, scaler = load_model(model_filename)  

    # Example input values (modify as necessary)  
    input_values = [4, 183, 67.0813, 21.38211, 0,28.4, 0.212,36]  

    # Check for sepsis and print the result  
    result = check_for_sepsis(classifier, scaler, input_values)  
    print(result)