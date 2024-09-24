import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print(f"========== GROUP 3: CASE STUDY - DATA PREPROCESSING ==========\n")

print(f"~ SECTION ON DESCRIPTIVE ANALYTICS AND VISUALIZATION ~\n")

# Loading the provided dataset
path = 'bankdata.csv'
df = pd.read_csv(path)

# Descriptive Statistics for Age, Income, and Children
print(f"--- NUMERICAL STATISTICS ---")
age_stats = df['age'].describe().round(4).to_frame(name="Age Stats")
income_stats = df['income'].describe().round(4).to_frame(name="Income Stats")
child_stats = df['children'].describe().round(4).to_frame(name="Children Stats")

# Display as DataFrames for better layout
print(f"--- Age Column Statistics ---")
print(age_stats)
print(f"\n--- Income Column Statistics ---")
print(income_stats)
print(f"\n--- Children Column Statistics ---")
print(child_stats)
print(f"\n{'-'*40}\n")  # Separator for better readability

# Relevant Age Values (Range, Min, Max, Median)
print(f"RELEVANT AGE VALUES: \n")
age_summary = pd.DataFrame({
    'Age Range': [df['age'].max() - df['age'].min()],
    'Min Age': [df['age'].min()],
    'Max Age': [df['age'].max()],
    'Median Age': [df['age'].median()]
})
print(age_summary)
print(f"\n{'-'*40}\n")  # Separator for better readability

# Customer segregation based on account type (savings or current)
print(f"--- SEGREGATION OF CUSTOMER ACCOUNT TYPE ---")
account_types = pd.DataFrame({
    'Savings Account Count': [df[df['save_act'] == 'YES'].shape[0]],
    'Current Account Count': [df[df['current_act'] == 'YES'].shape[0]]
})
print(account_types)
print(f"\n{'-'*40}\n")

# Pivot report on the relationship between Civil Status and Number of Children
print(f"--- PIVOT REPORT: MARRIED VS. NUMBER OF CHILDREN ---")
pivot_report = df.pivot_table(index='married', values='children', aggfunc='mean').round(4)
print(pivot_report)
print(f"\n{'-'*40}\n")

# Grouped means of Age, Income, and Children by PEP, Married, and Has Car
print(f"--- GROUPED MEANS ---")
grouped_means = df.groupby(['pep', 'married', 'car'])[['age', 'income', 'children']].mean().round(4)
print(grouped_means)
print(f"\n{'-'*40}\n")

# Pattern analysis for PEP purchase and Civil status (Age and PEP)
print(f"--- PATTERN ANALYSIS: AGE AND PEP ---")
pep_age_pattern = df.groupby('pep')['age'].mean().round(4).to_frame(name="Avg Age by PEP")
print(pep_age_pattern)
print(f"\n{'-'*40}\n")

# Pattern analysis: Number of Children vs. PEP and Marriage
print(f"--- PATTERN ANALYSIS: NUMBER OF CHILDREN, PEP, AND MARRIAGE ---")
pep_children_pattern = df.groupby(['pep', 'married'])['children'].mean().round(4).to_frame(name="Avg Children")
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
df['income_Binned'] = pd.qcut(df['income'], q=3, labels=["Low", "Medium", "High"])

print(f"--- BINNED INCOME COLUMN ---")
print(df[['income', 'income_Binned']].head())
print(f"\n{'-'*40}\n")

# Dummy variables for Region
print(f"--- DUMMY VARIABLES FOR REGION ---")
df_with_dummies = pd.get_dummies(df, columns=['region'], drop_first=False)
print(df_with_dummies.head())
print(f"\n{'='*50}\n")
