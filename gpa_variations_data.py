import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ai_student_impact_dataset.csv')
print(df.head())
print(df.columns.tolist())
print(df.dtypes)

print(df.shape)
print(df.isnull().sum())
print(df.info())
print(df.describe())

print(df['Major_Category'].value_counts())
print(df['Year_of_Study'].value_counts())
print(df['Primary_Use_Case'].value_counts())
print(df['Prompt_Engineering_Skill'].value_counts())
print(df['Burnout_Risk_Level'].value_counts())

df_ug = df[df['Year_of_Study'] != 'Graduate'] 

df_ug['GPA_Delta'] = df_ug['Post_Semester_GPA'] - df_ug['Pre_Semester_GPA'] #feature engineering
df_ug['AI_Usage_Level'] = pd.qcut(df_ug['Weekly_GenAI_Hours'], q=3, labels=['Light', 'Moderate', 'Heavy']) #feature engineering

gpa_by_usage = df_ug.groupby('AI_Usage_Level')['GPA_Delta'].mean() #analisi pronta per grafico 

plt.figure(figsize=(10, 6))

sns.barplot(
    x=gpa_by_usage.index,
    y=gpa_by_usage.values,
    hue=gpa_by_usage.index,
    palette='Blues_d',
    legend=False,
    width=0.8
)

plt.title('GPA Variation by AI Usage Level')
plt.xlabel('AI Usage Level')
plt.ylabel('GPA Variation (mean delta)')
plt.tight_layout()
plt.show()

gpa_by_major = df_ug.groupby(['Major_Category', 'AI_Usage_Level'])['GPA_Delta'].mean().reset_index() #analisi
plt.figure(figsize=(12, 6))

sns.barplot(
    data=gpa_by_major,
    x='Major_Category',
    y='GPA_Delta',
    hue='AI_Usage_Level',
    palette='Blues_d'
)

plt.title('GPA Variation by Major Category and AI Usage Level')
plt.xlabel('Major Category')
plt.ylabel('GPA Variation (mean delta)')
plt.legend(title='AI Usage Level')
plt.tight_layout()
plt.show()

burnout_data = df_ug.groupby(['AI_Usage_Level', 'Burnout_Risk_Level']).size().reset_index(name='count')

burnout_pivot = burnout_data.pivot(
    index='AI_Usage_Level',
    columns='Burnout_Risk_Level',
    values='count'
)

burnout_pivot[['Low', 'Medium', 'High']].plot(
    kind='bar',
    stacked=True,
    color=['#2ecc71', '#f39c12', '#e74c3c'],
    figsize=(10, 6)
)

plt.title('Burnout Risk Level by AI Usage Level')
plt.xlabel('AI Usage Level')
plt.ylabel('Number of Students')
plt.legend(title='Burnout Risk Level')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

for col in ['Anxiety_Level_During_Exams', 'Perceived_AI_Dependency', 'Skill_Retention_Score']:
    print(f"\n--- {col} ---")
    print(df_ug.groupby('AI_Usage_Level')[col].describe())

gpa_by_skill = df_ug.groupby(['Prompt_Engineering_Skill', 'AI_Usage_Level'])['GPA_Delta'].mean().reset_index()

skill_pivot = gpa_by_skill.pivot(
    index='AI_Usage_Level',
    columns='Prompt_Engineering_Skill',
    values='GPA_Delta'
)

plt.figure(figsize=(8, 5))

sns.heatmap(
    skill_pivot,
    annot=True,
    fmt='.3f',
    cmap='Blues',
    linewidths=0.5
)

plt.title('GPA Delta by Prompt Engineering Skill and AI Usage Level')
plt.xlabel('Prompt Engineering Skill')
plt.ylabel('AI Usage Level')
plt.tight_layout()
plt.show()