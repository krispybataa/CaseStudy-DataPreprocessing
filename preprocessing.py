import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print(f"GROUP 3: CASE STUDY - DATA PREPROCESSING")

print(f"~SECTION ON DECRIPTIVE ANALYTICS AND VISUALIZATION\n")
#Loading provided dataset
path = 'bankdata.csv'
df = pd.read_csv(path)

#Section on Descriptive Statistics
age_stats = df['age'].describe()
age_range = df['age'].max() - df['age'].min()

#Calculation on variance, sum, etc
age_var = df['age'].var()
age_sum = df['age'].sum()
age_count = df['age'].count()

#Printing of DS Section
print("Age Statistics: ")
print(age_stats)
print(f"Age Range: {age_range}")
print(f"Age Variance: {age_var}")
print(f"Age Sum: {age_sum}")
print(f"Age Count: {age_count}")


#Customer segregation based on account type (savings or current)
savings_acc_count = df[df['save_act'] == 'Yes'].shape[0]
current_acc_count = df[df['current_act'] == 'Yes'].shape[0]

print(f"Savings Account Count: {savings_acc_count}")
print(f"Current Account Count: {current_acc_count}")

#Pivot report on relationship between Civil Status and no. of children
pivot_report = df.pivot_table(index='married', values='children', aggfunc='mean')

print("Pivot report for Married vs. Number of Children:")
print(pivot_report)

#Calculation of means of specified attributes
grouped_means = df.groupby(['pep', 'married', 'car'])[['age', 'income', 'children']].mean()

print("Means of Age, Income, and Children by PEP, Married, and Has Car:")
print(grouped_means)


#Pattern analysis for PEP purchase and Civil status
pep_age_pattern = df.groupby('pep')['age'].mean()
pep_children_pattern = df.groupby(['pep', 'married'])['children'].mean()

print("Pattern of Age vs PEP:")
print(pep_age_pattern)
print("Pattern of Number of Children vs PEP and Marriage:")
print(pep_children_pattern)

print("\n~END OF DESCRIPTIVE ANALYTICS AND VISUALIZATION\n")

#########

print(f"~SECTION ON DATA TRANSFORMATION")

#Normalizing Income into [0,1] scale
scaler = MinMaxScaler()
df['income_Normalized'] = scaler.fit_transform(df[['income']])

print("Normalized Income Column:")
print(df[['income', 'income_Normalized']].head())

#Equal Depth (frequency)
df['income_Binned'] = pd.qcut(df['income'], q=3, labels=["Low", "Medium", "High"])

print("Binned Income Column:")
print(df[['income', 'income_Binned']].head())

#Dummy values for region
df_with_dummies = pd.get_dummies(df, columns=['region'], drop_first=False)

print("Dataframe with Dummy Variables for Region:")
print(df_with_dummies.head())