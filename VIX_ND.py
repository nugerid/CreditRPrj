import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_loan = pd.read_csv(r'E:\KULIAH\Kuliah\SEMESTER 8\Magang ID X Partners\Week 4\loan_data_2007_2014.csv',low_memory=False)
df_loan.head()
df_loan.drop(['number','out_prncp','out_prncp_inv','policy_code','annual_inc_joint','dti_joint','verification_status_joint','open_acc_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m','open_il_6m'],axis=1,inplace=True)

df_loan = df_loan[(df_loan['loan_status']=='Fully Paid') | (df_loan['loan_status']=='Charged Off')]
sns.countplot(x='loan_status', data=df_loan)
plt.show()

df_loan.corr()
plt.figure(figsize=(18,16))
sns.heatmap(df_loan.corr(), annot=False, cmap='viridis',linewidth=0.25)
plt.show()

df_loan.groupby('loan_status')['loan_amnt'].describe()
sns.boxplot(x='loan_status', y='loan_amnt', data=df_loan)
plt.show()

sns.countplot(x='grade', hue='loan_status', data=df_loan)
plt.show()

plt.figure(figsize=(12,4))
subgrade_order = sorted(df_loan['sub_grade'].unique())
sns.countplot(x= 'sub_grade', data=df_loan, order=subgrade_order, palette='coolwarm')
plt.show()

plt.figure(figsize=(12,4))
subgrade_order = sorted(df_loan['sub_grade'].unique())
sns.countplot(x= 'sub_grade', data=df_loan, order=subgrade_order, palette='coolwarm', hue='loan_status')
plt.show()

#Data Preprocessing
df_loan.head()
df_loan.isna().sum()

df_loan['emp_title'].value_counts()
df_loan = df_loan.drop('emp_title', axis=1)

sorted(df_loan['emp_length'].dropna().unique())
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']

plt.figure(figsize=(12,4))

sns.countplot(x='emp_length', data=df_loan, order=emp_length_order)
plt.figure(figsize=(12,4))

sns.countplot(x='emp_length', data=df_loan, order=emp_length_order, hue='loan_status')

emp_co = df_loan[df_loan['loan_status']== 'Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df_loan[df_loan['loan_status']== 'Fully Paid'].groupby('emp_length').count()['loan_status']

emp_len = emp_co/emp_fp
emp_len.plot(kind='bar')
plt.show()

df_loan = df_loan.drop('emp_length', axis=1)
df_loan.isna().sum()

df_loan['purpose'].head(10)
df_loan['title'].head(10)

df_loan = df_loan.dropna()
df_loan.isna().sum()

df_loan['term'].value_counts()
df_loan['term'] = df_loan['term'].apply(lambda term: int(term[:3]))
df_loan = df_loan.drop('grade', axis=1)
subgrade_dummies = pd.get_dummies(df_loan['sub_grade'], drop_first=True)
df_loan = pd.concat([df_loan.drop('sub_grade', axis=1), subgrade_dummies], axis=1)

dummies = pd.get_dummies(df_loan[['verification_status', 'application_type', 'initial_list_status', 'purpose']], drop_first = True)
df_loan = df_loan.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1)
df_loan = pd.concat([df_loan, dummies], axis=1)

df_loan['home_ownership'].value_counts()
df_loan['home_ownership'] = df_loan['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df_loan['home_ownership'], drop_first=True)
df_loan = df_loan.drop('home_ownership', axis=1)
df_loan = pd.concat([df_loan, dummies], axis=1)
df_loan = df_loan.drop('issue_d', axis=1)
df_loan['earliest_cr_year'] = df_loan['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df_loan = df_loan.drop('earliest_cr_line', axis=1)

df_loan = df_loan.drop('loan_status', axis=1)

#Prediction Model
#Saya belum mampu membangun model