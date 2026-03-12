import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('sa_financial_data.csv')

# Explore data
print(df.head())
print(df.shape)
print(df.dtypes)
print("Missing values:\n", df.isna().sum())

# Clean data
df = df.drop_duplicates()
df['gender'] = df['gender'].map({0: 'Male', 1: 'Female'})

# General Analysis 
for col in ['income_bracket', 'province', 'education', 'gender']:
    print(f"\n── Savings Rate by {col} ──")
    print(df.groupby(col)['has_savings'].mean().mul(100).round(1))
    print(f"\n── Credit Rate by {col} ──")
    print(df.groupby(col)['has_credit'].mean().mul(100).round(1))

# Q1: Do high-income individuals save more? 
savings_by_income = df.groupby('income_bracket')['has_savings'].mean().mul(100).round(1)

plt.figure(figsize=(7, 4))
plt.bar(savings_by_income.index, savings_by_income.values, color='steelblue', edgecolor='white')
plt.title('Savings Rate by Income Bracket')
plt.ylabel('% with Savings')
plt.xlabel('Income Bracket')
plt.tight_layout()
plt.show()

# Q2: Are low-income individuals less likely to have credit? 
credit_by_income = df.groupby('income_bracket')['has_credit'].mean().mul(100).round(1)

plt.figure(figsize=(7, 4))
plt.bar(credit_by_income.index, credit_by_income.values, color='darkorange', edgecolor='white')
plt.title('Credit Access by Income Bracket')
plt.ylabel('% with Credit')
plt.xlabel('Income Bracket')
plt.tight_layout()
plt.show()

# Q3: Middle-income with BOTH credit and savings 
middle = df[df['income_bracket'] == 'Middle']

if middle.empty:
    print("No middle-income individuals found.")
else:
    both = middle[(middle['has_savings'] == 1) & (middle['has_credit'] == 1)]
    pct = len(both) / len(middle) * 100
    print(f"\nMiddle-income with both credit & savings: {pct:.1f}%")

# Q4: Does education level affect credit access? 
credit_by_edu = df.groupby('education')['has_credit'].mean().mul(100).round(1).sort_values()

plt.figure(figsize=(8, 4))
plt.barh(credit_by_edu.index, credit_by_edu.values, color='seagreen', edgecolor='white')
plt.title('Credit Access by Education Level')
plt.xlabel('% with Credit')
plt.ylabel('Education Level')
plt.tight_layout()
plt.show()

# Q5: Does education level affect savings? 
savings_by_edu = df.groupby('education')['has_savings'].mean().mul(100).round(1).sort_values()

plt.figure(figsize=(8, 4))
plt.barh(savings_by_edu.index, savings_by_edu.values, color='mediumpurple', edgecolor='white')
plt.title('Savings Rate by Education Level')
plt.xlabel('% with Savings')
plt.ylabel('Education Level')
plt.tight_layout()
plt.show()

# Q6: Gender gap in credit and savings
gender_summary = df.groupby('gender')[['has_credit', 'has_savings']].mean().mul(100).round(1)

genders = gender_summary.index.tolist()
credit_vals = gender_summary['has_credit'].values
savings_vals = gender_summary['has_savings'].values

x = range(len(genders))
width = 0.35

plt.figure(figsize=(7, 4))
plt.bar([i - width/2 for i in x], credit_vals, width=width, label='% with Credit', color='steelblue', edgecolor='white')
plt.bar([i + width/2 for i in x], savings_vals, width=width, label='% with Savings', color='seagreen', edgecolor='white')
plt.xticks(ticks=x, labels=genders)
plt.title('Credit & Savings by Gender')
plt.ylabel('% of Group')
plt.xlabel('Gender')
plt.legend()
plt.tight_layout()
plt.show()

# Q7: Education x Income — savings heatmap 
pivot = df.pivot_table(values='has_savings', index='education',
                       columns='income_bracket', aggfunc='mean').mul(100).round(1)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot.values, cmap='YlGnBu', aspect='auto')

ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels(pivot.columns)
ax.set_yticklabels(pivot.index)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.values[i, j]:.1f}%", ha='center', va='center', fontsize=10, color='black')

plt.colorbar(im, ax=ax, label='% with Savings')
plt.title('Savings Rate (%) by Education & Income Bracket')
plt.tight_layout()
plt.show()

# Bonus: Financial vulnerability (no credit AND no savings) 
df['vulnerable'] = ((df['has_credit'] == 0) & (df['has_savings'] == 0)).astype(int)
vuln_by_province = df.groupby('province')['vulnerable'].mean().mul(100).round(1).sort_values()

plt.figure(figsize=(9, 5))
plt.barh(vuln_by_province.index, vuln_by_province.values, color='firebrick', edgecolor='white')
plt.title('Financially Vulnerable Population by Province')
plt.xlabel('% with No Credit & No Savings')
plt.tight_layout()
plt.show()
