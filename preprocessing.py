import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#VISUALIZATION VIA MATPLOTLIB
import matplotlib.pyplot as plt

print(f"========== GROUP 3: CASE STUDY - DATA PREPROCESSING ==========\n")

print(f"~ SECTION ON DESCRIPTIVE ANALYTICS AND VISUALIZATION ~\n")

# Loading the provided dataset
path = 'bankdata.csv'
df = pd.read_csv(path)

# Descriptive Statistics for Age
print(f"--- NUMERICAL STATISTICS ---")
age_stats = df['age'].describe().round(4)
income_stats = df['income'].describe().round(4)
child_stats = df['children'].describe().round(4)

age_min = df['age'].min()
age_max = df['age'].max()
age_range = age_max - age_min

# Calculation for variance, sum, etc.
age_med = df['age'].median()
age_sum = df['age'].sum()
age_count = df['age'].count()

# Printing Descriptive Statistics Section
print("Age Column: ")
print(age_stats)
print("\nIncome Column: ")
print(income_stats)
print("\nChildren Column: ")
print(child_stats)
print(f"\n{'-'*40}\n")  # Separator for better readability

print(f"RELEVANT AGE VALUES: \n")
print(f"Age Range: {age_range}")
print(f"Minimum Age: {age_min}")
print(f"Maximum Age: {age_max}")
print(f"Average Age: {age_med}")
print(f"\n{'-'*40}\n")  # Separator for better readability

# Customer segregation based on account type (savings or current)
print(f"--- SEGREGATION OF CUSTOMER ACCOUNT TYPE ---")
savings_acc_count = df[df['save_act'] == 'YES'].shape[0]
current_acc_count = df[df['current_act'] == 'YES'].shape[0]

print(f"Savings Account Count: {savings_acc_count}")
print(f"Current Account Count: {current_acc_count}")
print(f"\n{'-'*40}\n")

# Pivot report on the relationship between Civil Status and Number of Children
print(f"--- PIVOT REPORT: MARRIED VS. NUMBER OF CHILDREN ---")
pivot_report = df.pivot_table(index='married', values='children', aggfunc='mean').round(4)
print(pivot_report)
print(f"\n{'-'*40}\n")

# Calculation of means of Age, Income, and Children by PEP, Married, and Has Car
print(f"--- GROUPED MEANS ---")
grouped_means = df.groupby(['pep', 'married', 'car'])[['age', 'income', 'children']].mean().round(4)
print(grouped_means)
print(f"\n{'-'*40}\n")

# Pattern analysis for PEP purchase and Civil status
print(f"--- PATTERN ANALYSIS: AGE AND PEP ---")
pep_age_pattern = df.groupby('pep')['age'].mean().round(4)
print(pep_age_pattern)

print(f"\n--- PATTERN ANALYSIS: NUMBER OF CHILDREN, PEP, AND MARRIAGE ---")
pep_children_pattern = df.groupby(['pep', 'married'])['children'].mean().round(4)
print(pep_children_pattern)
print(f"\n{'-'*40}\n")

print(f"~ END OF DESCRIPTIVE ANALYTICS AND VISUALIZATION ~\n")

#########

print(f"~ SECTION ON DATA TRANSFORMATION ~\n")

# Normalizing Income into [0,1] scale
scaler = MinMaxScaler()
df['income_Normalized'] = scaler.fit_transform(df[['income']]).round(4)

print(f"--- NORMALIZED INCOME COLUMN ---")
print(df[['income', 'income_Normalized']].head())
print(f"\n{'-'*40}\n")

# Equal Depth (frequency) binning
df['income_Binned'] = pd.qcut(df['income'], q=3, labels=["Low", "Medium", "High"]).round(4)

print(f"--- BINNED INCOME COLUMN ---")
print(df[['income', 'income_Binned']].head())
print(f"\n{'-'*40}\n")

# Dummy values for Region
print(f"--- DUMMY VARIABLES FOR REGION ---")
df_with_dummies = pd.get_dummies(df, columns=['region'], drop_first=False)
print(df_with_dummies.head())
print(f"\n{'='*50}\n")