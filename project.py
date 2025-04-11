import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data
data = pd.read_csv("D:\\Downloads\\Clean_Python_Project 1.csv")
print(data)
print(data.isnull().sum())

# Save cleaned version
data.to_csv("Clean_File1.csv", index=False)

# Load the cleaned data
df = pd.read_csv("Clean_File1.csv")

# 2. Boxplot by Age Group
plt.figure(figsize=(12,6))
age_group = df[df['Group'] == 'By Age']
sns.boxplot(data=age_group, x='Subgroup', y='Clean_Value', hue='Indicator')
plt.title("Distribution of Symptoms by Age Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Barplot by State (Latest Phase)
latest_phase = df['Phase'].dropna().max()
latest_data = df[(df['Phase'] == latest_phase) & 
                 (df['Indicator'] == 'Symptoms of Anxiety Disorder') & 
                 (df['Group'] == 'By State')]
plt.figure(figsize=(14,7))
sns.barplot(data=latest_data, x='State', y='Clean_Value')
plt.xticks(rotation=90)
plt.title(f"Anxiety Symptoms by State (Latest Phase: {latest_phase})")
plt.tight_layout()
plt.show()

# 4. KDE Plot
plt.figure(figsize=(10,5))
sns.kdeplot(data=df, x='Clean_Value', hue='Indicator', fill=True)
plt.title("Density Plot of Symptom Percentages")
plt.tight_layout()
plt.show()

# 5. Violin Plot by Gender
if 'By Sex' in df['Group'].unique():
    gender_group = df[df['Group'] == 'By Sex']
    plt.figure(figsize=(10,6))
    sns.violinplot(data=gender_group, x='Subgroup', y='Clean_Value', hue='Indicator')
    plt.title("Symptoms by Gender")
    plt.tight_layout()
    plt.show()

# 6. Lineplot of Anxiety by Age Group Over Time
plt.figure(figsize=(12,6))
age_trend = df[(df['Group'] == 'By Age') & (df['Indicator'] == 'Symptoms of Anxiety Disorder')]
sns.lineplot(data=age_trend, x='Time Period Start Date', y='Clean_Value', hue='Subgroup', marker='o')
plt.title("Anxiety Symptoms Over Time by Age Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Heatmap using coolwarm
heatmap_df = df[df['Group'] == 'By State'].groupby(['State', 'Indicator'])['Clean_Value'].mean().unstack()

top_states = heatmap_df.mean(axis=1).sort_values(ascending=False).head(15).index
heatmap_top = heatmap_df.loc[top_states]

plt.figure(figsize=(16,9))
sns.heatmap(heatmap_top.round(1), annot=True, fmt=".1f", cmap='coolwarm', linewidths=0.5, linecolor='gray', cbar_kws={"label": "Average Symptom (%)"})
plt.title("Top 15 States: Average Reported Symptoms by Indicator", fontsize=14)
plt.xlabel("Indicator")
plt.ylabel("State")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# 8. Additional Barplot: National Comparison of Symptoms
latest_date = df['Time Period Start Date'].dropna().max()
latest_national = df[(df['Group'] == 'National Estimate') & (df['Time Period Start Date'] == latest_date)]

if not latest_national.empty:
    plt.figure(figsize=(8,5))
    sns.barplot(data=latest_national, x='Indicator', y='Clean_Value', palette='Set2')
    plt.title(f"National Comparison of Symptoms (as of {latest_date})")
    plt.ylabel("Reported Symptoms (%)")
    plt.tight_layout()
    plt.show()
else:
    print("No national estimate data available for the latest date.")

# 10. Facet Grid: Depression in Top 6 States
top_states = df[df['Group'] == 'By State'].groupby('State')['Clean_Value'].mean().sort_values(ascending=False).head(6).index
facet_data = df[(df['Group'] == 'By State') & 
                (df['State'].isin(top_states)) & 
                (df['Indicator'] == 'Symptoms of Depression Disorder')]
if not facet_data.empty:
    g = sns.FacetGrid(facet_data, col='State', col_wrap=3, height=4, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='Time Period Start Date', y='Clean_Value')
    g.set_titles("{col_name}")
    g.set_axis_labels("Date", "Depression Symptom (%)")
    plt.suptitle("Depression Trend Over Time for Top 6 States", y=1.05)
    plt.tight_layout()
    plt.show()
else:
    print("No data available for depression symptoms in top states.")

# 11. Pairplot using coolwarm
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
pair_data = df[numeric_cols].dropna().sample(n=500, random_state=1)

sns.set(style="whitegrid")
pair_plot = sns.pairplot(
    pair_data,
    corner=True,
    palette='coolwarm',
    height=2.8,
    aspect=1.2,
    plot_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'k'},
    diag_kws={'fill': True, 'color': 'lightcoral'}
)
pair_plot.fig.suptitle("Pairplot of All Numerical Features", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# Save final modified file
df.to_csv("Modified_File.csv", index=False)
