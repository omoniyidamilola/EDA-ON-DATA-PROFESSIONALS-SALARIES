# EDA-ON-DATA-PROFESSIONALS-SALARIES
This project was done using python.
# EDA ON DATA PROFESSIONALS SALARIES

## About the dataset

The data was downloaded from Kaggle datasets: https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023/download?datasetVersionNumber=1

Data Science Job Salaries  
Dataset contains 11 columns, each are: 



1.  work_year: The year salary was paid.



2.  experience_level: The experience level in the job during the year.
      
      
      
3.  employment_type: The type of employment for the role.
      
      
      
4.  job_title: The role worked in during the year.



5.  salary: The total gross salary amount paid.



6.  salary_currency: The currency of the salary paid as an ISO 4217 currency code.



7.  salary_in_usd: The salary in USD.



8.  employee_resisdence: Employee's primary country of resisdence during the work year as an ISO 3166 country code(Alpha-2 code).



9.  remote_ratio: The overall amount of work done remotely:
        (a) 0 = No remote work (less than 20%),
        (b) 50 = Partially remote,
        (c) 100 = Fully remote(more than 80%).
        
        
        
10. company_location: The country of the employer's main office or contracting branch.



11. company_size: The average number of people that worked for the company during the year.
        (a) S (small) = less than 50 employees,
        (b) M (medium) = 50-250 employees ,
        (c) L (large) = more than 250 employees.

### Problem Statement

#### To Identify:
    
(1) Which Experience level has the highest hiring?

(2) Which Employment type does company hire most?

(3) Which Job Title has the highest pay?

(4) Does the Company size affect the rate of hiring and pay.

(5) Where are the most employee residences
 

## Importing Libraries

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
'%matplotlib inline'
import seaborn as sns

import plotly.express as px

import nltk
!pip install nltk

!pip install wordcloud
from wordcloud import WordCloud

#Load dataset
salary = pd.read_csv('ds_salaries.csv')

salary

#Check column information
salary.info()

#Summary statistics of dataset
salary.describe()

#lets get the shape of the dataset
salary.shape

#Check for columns label
salary.columns

salary.rename(columns ={'work_year':'Work_year', 'experience_level':'Experience_level', 'employment_type':'Employment_type','job_title':'Job_title', 'salary':'Salary','salary_currency':'Salary_currency', 'salary_in_usd':'Salary_in_usd', 'employee_residence':'Employee_residence', 'remote_ratio':'Remote_ratio', 'company_location':'Company_location', 'company_size':'Company_size' }, inplace=True)
#renaming the columns name.

salary

## Data Cleaning

##### .isna() is used to check for the null values.

salary.isna()

##### .isna().sum() is used to count for the number of null values in each column

salary.isna().sum()

#Checking if there is duplicate values
salary.duplicated()

#Checking for the total number of duplicate values.
salary.duplicated().sum()

##### .drop_duplicates() is used to drop duplicate

salary.drop_duplicates()

salary.drop_duplicates(inplace=True)

salary

## Exploratory Data Analysis

### Experience Level

counts= salary.Experience_level.value_counts()
counts

### SE - Senior Expertise,  MI - Mid -Intermediate, EN- Entry level, EX- Executive

plt.figure(figsize=(8, 5))

counts = salary['Experience_level'].value_counts()
sns.barplot(x=counts.values, y=counts.index)

plt.title('Experience Level ')
plt.xlabel('counts')
plt.ylabel('Experience Level')
plt.show()

####  From the above bar chart, Senior expertise is higher by counts, senior expertise mostly requires experience. Senior expertise are the highest hire maybe because of their experience level.

#### Salaries of Employees Based on Experience Level

labels = {'Experience_level':'experience_level', 'Salary_in_usd':'salary_in_usd'}

font1= {'family':'serif', 'color':'black', 'size':15}
font2= {'family':'serif', 'color':'red', 'size':15}

px.box(salary, x = 'Experience_level', y= 'Salary_in_usd', color = 'Experience_level',labels=labels, title='<b>Salary by Experience Level')

#### Executive has more pay by salary compared to mid-intermediate, entry-level and senior expertise.

### Job Title

print('Total Job title:', salary['Job_title'].value_counts().size)

job_title_count=salary['Job_title'].value_counts()
job_title_count


top_20_job_titles = job_title_count[:20]

plt.figure(figsize=(12, 8))
sns.barplot(x=top_20_job_titles.values, y=top_20_job_titles.index)


font1= {'family':'serif', 'color':'black', 'size':15}
font2= {'family':'serif', 'color':'red', 'size':15}


plt.title('Top 20 Job Titles', fontdict=font1)
plt.xlabel('counts', fontdict=font2)
plt.ylabel('Job Title', fontdict=font2)
plt.show()

#### Data Engineer, Data Scientist and Data Analyst are the top 3 job title though other job titles are also related to the 3 job titles. 


## WordCloud of Job Title

def Freq_salary(cleanwordlist):
    Freq_dist_nltk = nltk.FreqDist(cleanwordlist)
    salary_freq = pd.DataFrame.from_dict(Freq_dist_nltk, orient='index', columns=['Frequency'])
    salary_freq.index.name = 'Term'
    salary_freq.reset_index(inplace=True)  # Add this line to include the 'Term' column
    return salary_freq

def Word_Cloud(data, color_background, colormap, title):
    
    plt.figure(figsize = (20,15))
    
    wc= WordCloud(width=1200, height=600, max_words = 50, colormap = colormap, max_font_size =100, random_state =88, background_color = color_background).generate_from_frequencies(data)
    
    plt.imshow(wc, interpolation = 'bilinear')
    
    plt.title(title, fontsize = 22)
    
    plt.axis('off')
    
    plt.show()

freq_salary = Freq_salary(salary['Job_title'].values.tolist())

data = dict(zip(freq_salary['Term'].tolist(), freq_salary['Frequency'].tolist()))

data = freq_salary.set_index('Term').to_dict()['Frequency']


Word_Cloud(data, 'black', 'RdBu', 'WordCloud of Job Title')

### Salary Distribution

salaryusd= salary['Salary_in_usd'].value_counts()
salaryusd

The company pay 58 employees ($100,000)

56 employees($150,000), 51 employees($120,000), 

47 employees($200,000),  39 employees($130,000),   

$314,100  is paid to just an employee,  

$195,800  is paid to just an employee,  

$262,500  is paid to just an employee,

$209,450  is paid to just an employee, 

$94,665   is also paid to just an employee.

first_15_salary_usd=salaryusd[ :15]

first_15_salary_usd.plot(kind= 'bar', color='purple')

font1= {'family':'serif', 'color':'black', 'size':20}
font2= {'family':'serif', 'color':'red', 'size':15}

plt.title('Highest Earn', fontdict=font1)
plt.xlabel('Salary in usd', fontdict=font2)
plt.ylabel('Count', fontdict=font2)
plt.show()

sns.histplot(salary['Salary_in_usd']/1000,
bins = 30, kde=True)


plt.title ('Salary Distribution')

plt.xlabel ('Salary_in_usd')

plt.ylabel ('Employee')

plt.minorticks_on()

#### The salary distribution is positively skewed

salary_1=salary['Salary_currency'].value_counts()
salary_1

first_10_salary_currency = salary_1[ :8]

first_10_salary_currency.plot(kind = 'barh', color='green')

font1= {'family':'serif', 'color':'black', 'size':20}
font2= {'family':'serif', 'color':'red', 'size':15}

plt.title('Highest Currency Earn', fontdict=font1)
plt.xlabel('Count', fontdict=font2)
plt.ylabel('Salary Currency', fontdict=font2)
plt.show()

#### The company pays the employees  more  in USD compared to the other currency.

new_salary = salary.groupby('Job_title', as_index = False) ['Salary_in_usd'].mean().tail()
new_salary

DF = salary.groupby('Job_title', as_index = False) ['Salary_in_usd'].mean().head()
DF

## Company Size

salary['Company_size'].value_counts()

#### S = Small             M = Medium                  L = Large  

company = salary['Company_size'].value_counts()
label = company.index
sizes = company.values
color = ['magenta', 'orange', 'green']
explode = (0.2,0.1,0)

font1= {'family':'serif', 'color':'red', 'size':10}
font2= {'family':'serif', 'color':'purple', 'size':15}

plt.figure(figsize = (4,6))
plt.pie(company, labels = label, autopct = '%1.1f%%', startangle = 90, colors = color, explode = explode, shadow = True)
plt.title('A Pie Chart Showing The Distribution of Company size', fontdict=font1)
plt.legend(title = 'company', loc = 'best')
plt.axis('equal')
plt.show()

#### Medium Firm is higher than the other firm by 78.5%, large Firm is 15.8% while Small Firm is 5.7%.

### Company size By Remote Ratio

plt.figure(figsize=(10, 5))

font1= {'family':'serif', 'color':'black', 'size':15}
font2= {'family':'serif', 'color':'red', 'size':15}

sns.set_palette('Set1')  
ax = sns.countplot(data=salary, x='Company_size', hue='Remote_ratio')
ax.set_title('Company Size By Remote Ratio')

plt.show() 

#### Medium firm provide maximum onsite opportunities and minimum fully remote work.

### Work Year

company = salary['Work_year'].value_counts()
label = company.index
sizes = company.values
color = ['purple', 'yellow', 'hotpink', 'red']
explode = (0.2, 0.2, 0.3, 0)

font1= {'family':'serif', 'color':'wine', 'size':20}
font2= {'family':'serif', 'color':'red', 'size':15}

plt.figure(figsize = (4,6))
plt.pie(company, labels = label, autopct = '%1.1f%%', startangle = 90, colors = color, explode = explode, shadow = True)
plt.title('Distribution of Work Year', fontdict=font2, loc='center')
plt.legend( title = 'company',loc = 'upper left')
plt.axis('equal')
plt.show()

#### The company has more employees in 2023 about 44.7% than the other work year while 2022 has 43.5%. 2021 has 8.8% .  2020 has 2.9%: Decrease in 2020 maybe due to COVID pandemic.

sns.countplot(x = 'Experience_level', hue = 'Work_year', data= salary)

font1= {'family':'serif', 'color':'black', 'size':12}
font2= {'family':'serif', 'color':'red', 'size':15}

plt.title('Experience Level by Work Year', fontdict=font1)
plt.xlabel('Experience Level', fontdict=font2)
plt.ylabel('Count', fontdict=font2)
plt.show()

#### From the above chart, There is an increase in the number of senior expertise employed in 2023 but low intake of senior expertise in 2020, there was low intake of executive in 2020,2021,2022 and 2023.

### Company Location

salary['Company_location'].value_counts()

### Employment Type

Employment_type = salary.Employment_type.value_counts()
Employment_type

#### FT: (FULL-TIME) ,          CT: (CONTRACT) ,     FL: (FREELANCE),        PT: (PART-TIME). 

sns.countplot(x = 'Employment_type',data= salary)

font1= {'family':'serif', 'color':'black', 'size':15}
font2= {'family':'serif', 'color':'red', 'size':15}

plt.title('Employment Type', fontdict=font1)
plt.xlabel('Employment', fontdict=font2)
plt.ylabel('counts', fontdict=font2)
plt.show()

#### From the above chart, the employees are mostly full time staff. The company rarely take in contract staff, freelance staff and part-time staff. The company prefers to hire full time employee maybe because of their commitment to work.

sns.countplot(x = 'Employment_type', hue='Work_year', data= salary)

font1= {'family':'serif', 'color':'black', 'size':12}
font2= {'family':'serif', 'color':'red', 'size':15}

plt.title('Chart Showing Employment Type by Year', fontdict=font1)
plt.xlabel('Employment type', fontdict=font2)
plt.ylabel('count', fontdict=font2)
plt.show()

#### From the above chart,  There was an increase in the number of full-time staff in 2023 compared to 2020. 

### Employee Residence

Employee_residence = salary.Employee_residence.value_counts()
Employee_residence

Employee_residence = salary.Employee_residence.value_counts()

Employee_residence = Employee_residence[:5]
Employee_residence.plot(kind = 'bar')

font1= {'family':'serif', 'color':'black', 'size':15}
font2= {'family':'serif', 'color':'red', 'size':15}


plt.title('Top 5 Employee Residence', fontdict=font1)
plt.xlabel('counts', fontdict=font2)
plt.ylabel('Employee Residence', fontdict=font2)
plt.show()

#### Most of the employees are domiciled in the USA  probably because USA have much higher wages than the other countries

### Remote Ratio

Remote_ratio = salary.Remote_ratio.value_counts()
Remote_ratio

labels= ['Fully Remote', 'On-site', 'Partially Remote']

plt.bar(x = labels, height = Remote_ratio.values, width = 0.3)
Remote_ratio= salary.Remote_ratio.value_counts()

font1= {'family':'serif', 'color':'black', 'size':15}
font2= {'family':'serif', 'color':'red', 'size':15}



plt.title('Remote Work', fontdict=font1)

plt.xlabel('Remote ratio', fontdict=font2)
plt.ylabel('Frequency', fontdict=font2)

plt.show()

#### From the chart, the employee works 0% of the time or 100% of the time compared to those that work 50%. This could mean that the company are either adopting a Fully remote or On-site culture rather than a Hybrid approach.  

### Salaries of Employees Based on Employment Type

labels = {'Employment_type':'employment_type', 'Salary_in_usd':'salary_in_usd'}

font1= {'family':'serif', 'color':'black', 'size':15}
font2= {'family':'serif', 'color':'red', 'size':15}

px.box(salary, x = 'Employment_type', y= 'Salary_in_usd', color = 'Employment_type',labels=labels, title='<b>Salary by Employment type')

#### Full time has better salaries than contract, freelance, and part-time maybe because they are more committed to their work compared to the other employment type.

### Average Salary By Employment type

salary.groupby('Employment_type')['Salary_in_usd'].mean()

### Salaries of Employees Based on Company Size

salaryinusd_by_company_size = salary.groupby(['Salary_in_usd', 'Company_size']).size().reset_index(name='count')
top_10_companysize = salaryinusd_by_company_size.groupby('Company_size').apply(lambda x: x.nlargest(10, 'count')).reset_index(drop=True)

print('Salary in USD by Company Size:')
print(top_10_companysize)

#### The medium firm pays more salary than Large and small firms. This could be due to the fact that the large firm has more financial commitment to the other sectors owing to their large size or simply put they have more employees to pay.
    

### Grouping Job Title and Experience Level 

pd.options.display.max_rows = 999

salary = pd.read_csv('ds_salaries.csv')
jobtitle_by_experiencelevel = salary.groupby(['job_title', 'experience_level']).size().reset_index(name='count')
top_10_experiencelevel = jobtitle_by_experiencelevel.groupby('experience_level').apply(lambda x: x.nlargest(10, 'count')).reset_index(drop=True)

print('Job Titles by Experience Level:')
print(top_10_experiencelevel)

#### Data engineers seems to have more of entry-level, mid-intermediate, executive and senior expertise than Data Scientist and Data Analyst. 

fig, ax = plt.subplots(figsize=(20, 10))

numeric_columns = salary.select_dtypes(include=['float64', 'int64'])

# Selecting only numeric columns
sns.heatmap(numeric_columns.corr(), vmax=0.8, square=True, annot=True)

plt.title('Correlation Matrix', fontsize=15)

plt.show()

### Grouping Work Year and Remote Ratio

pd.options.display.max_rows = 999

salary = pd.read_csv('ds_salaries.csv')
workyear_by_remoteratio = salary.groupby(['work_year', 'remote_ratio']).size().reset_index(name='count')
top_10_workyear = workyear_by_remoteratio.groupby('work_year').apply(lambda x: x.nlargest(10, 'count')).reset_index(drop=True)

print('Work year by Remote ratio:')
print(top_10_workyear)

#### In 2020, 2021 and 2022, Employees that works fully remote are higher this maybe due to COVID-19 pandemic changing the work trend. BUT in 2023 employees mostly works onsite.

### Grouping Job Title and Experience Level¶

top_10_job_titles = salary['job_title'].value_counts().nlargest(10).index

plt.figure(figsize=(12, 12))
sns.boxplot(x=salary[salary['job_title'].isin(top_10_job_titles)]['salary_in_usd'],
            y=salary[salary['job_title'].isin(top_10_job_titles)]['job_title'],
            showfliers=True)
plt.title('Relationship between Salary and Top 10 Job title')
plt.show()

result = salary.groupby('job_title').agg(salary=('salary_in_usd', 'sum'))

result = result.sort_values('salary', ascending=False)

print(result)

#### Data Engineer has the highest pay by salary

### Insights From the Report

Based on Experience level: Senior expertise is higher by counts, Senior expertise requires experience. Executive has more pay by salary compared to mid-intermediate, entry-level and senior expertise.  
    
Based on Job Title: Total job title is 93, Data Engineer, Data Scientist and Data Analyst are the top 3 job title though other job titles are also related to those 3 job titles. 
    
Based on Salary: The salary distribution is positively skewed. The company pays the employees more in USD compared to the other currency.         
    
Based on Company size: Medium Firm is higher than the other firm by 78.5%, large Firm is 15.8% while Small Firm is 5.7%.  Medium companies provide maximum onsite opportunities and minimum fully remote work.                                                                          
The company has more employees in 2023 about 44.7% than the other work year while 2022 has 43.5%. 2021 has 8.8% .  2020 has 2.9%. Decrease in 2020 maybe due to the impact of COVID-19 pandemic.                   
    
Experience Level by Work year:There is an increase in the number of senior expertise employed in 2023 but low intake of senior expertise in 2020, there was low intake of executive in 2020,2021,2022 and 2023.   
    
Based on Employment type: The employees are mostly full time staff. The company rarely take in contract staff, freelance staff and part-time staff  . 
There was an increase in the number of full-time staff in 2023 compared to 2020.   

Based on Employee residence: Most of the employees are domiciled in USA probably because USA have much higher wages than other countries.

Based on Remote ratio: The employee works 0% of the time or 100% of the time compared to those that work 50%. This could mean that the company are either adopting a Fully remote or On-site culture rather than a Hybrid approach.  In 2020,Employees that works fully remote are higher, same applies to 2021 this maybe due to COVID-19 pandemic changing the work trend. In 2022, Employees that works fully remote are higher BUT in 2023 employees mostly works onsite.   

Data engineers seems to have more of entry-level, mid-intermediate, executive and senior expertise than Data Scientist and Data Analyst.  Full time has more pay than contract, freelance, part-time this maybe due to their committment. The medium firm pays more salary than Large and small firms. This could be due to the fact that the large firm has more financial commitment to the other sectors owing to their large size or simply put they have more employees to pay.
    
                                                             

### RECOMMENDATIONS

Company should focus on hiring and retaining professionals in the top three job titles: Data Engineer, Data Scientist and Data Analyst. These roles seem to be in high demand and are crucial to the company's operations. Consider providing career development opportunities and competitive compensation packages to attract and retain talent in these roles.
    
Company should consider conducting a salary review to ensure fair compensation across all job levels and experience levels. Address any significant discrepancies and provide salary adjustments where necessary to promote employee satisfaction and retention.

The majority of employees work in medium-sized firms and these firms provide the most onsite opportunities. Company should consider leveraging the benefits of being a medium-sized company, such as a close-knit work environment and potential for career growth, to attract and retain employees.

Implementation of strategies to maintain a steady workforce growth.

While Full-time staff is the dominant employment type, it's important to ensure that other types of employment, such as contract, freelance and part-time staff are adequately considered when needed. Evaluation of flexibility and benefits provided to different employment types to maintain a diverse and agile workforce.

Company should consider offering additional perks, such as flexible working hours, health and wellness programs and professional development allowances, to attract and retain top talent.

Majority of the employees reside in the USA, company should consider expanding recruitment efforts to diversify the employee base.

Although company seems to have a preference for either fully remote or onsite work, it is important to consider implementing a hybrid approach that allows flexibility and accomodates employee prerferences. Conduct surveys or gather feedback to assess the feasibility and desirability of a hybrid work model within the company.

Implement initiatives to enhance employee engagement and wellbeing. This can include fostering a positive work culture, promoting work-life balance, providing mental health resourses and encouraging open communication channels within the organization.



