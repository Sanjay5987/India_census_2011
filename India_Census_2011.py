#!/usr/bin/env python
# coding: utf-8

# #                                        Census of India(2011)

# ## Libraries
# Python libraries like ```pandas```, ```numpy```, ```matplotlib```, ```seaborn``` are used to do the Exploratory Data Analysis over the data
# and for making a good visualization out of it

# In[35]:


# Importing required libraries

get_ipython().system('pip install pandas --upgrade  ')
get_ipython().system('pip install numpy --upgrade  --q            ')
get_ipython().system('pip install matplotlib                   ')
get_ipython().system('pip install seaborn --upgrade          ')
get_ipython().system('pip install jovian --upgrade --quiet')

import warnings
warnings.filterwarnings('ignore')


# ```Jovian``` is the warehouse where you can save your notebook ```warnings``` library is used to hide any unwanted warnings

# In[36]:


#Importing installed libraries

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import pandas as pd
import os
import numpy as np
import jovian


# ### Dataset
# **The datset used here is the *Last census of India* that took place in 2011**

# In[37]:


census_dataframe = pd.read_csv('india-districts-census-2011.csv')  # Loading dataset and assigning vector to it
census_df = census_dataframe.copy()                                # Creating a copy of the dataset that will be used further in Analysis


# ## Exploring Dataset

# In[38]:


census_df.shape                         # It gives a view of number of rows and number of columns in the dataset


# In[39]:


census_df.describe()                      # It gives a quick look at various arithemetic calculations of different numeric columns, calculations like count, mean, median etc.


# In[40]:


census_df.info()


# In[41]:


census_df.sample(5)                       # '''sample''' function can be used to get a view of random data from the datasetdata 


# In[42]:


census_df.columns


# ## Filtering 

# #### Extracting required columns out of Dataset

# In[43]:


census_ind = census_df[['State name','District name','Population','Male','Female','Literate','Male_Literate','Female_Literate','SC','Male_SC','Female_SC','ST','Hindus','Muslims','Christians','Sikhs','Buddhists','Jains','Rural_Households','Urban_Households','Households','Below_Primary_Education','Primary_Education','Middle_Education','Secondary_Education','Higher_Education','Age_Group_0_29','Age_Group_30_49','Age_Group_50','Age not stated'
]]

census = census_ind.copy()


# ##### Correcting column names and making new dataframes for analysis

# In[44]:


census.rename(columns = {'State name':'state_name', 'District name':'district_name', 'Age not stated':'Age_not_stated'}, inplace = True)
census_ind_df = census.groupby('state_name')[['Population','Male','Female','Literate','Male_Literate','Female_Literate','SC','Male_SC','Female_SC','ST','Hindus','Muslims','Christians','Sikhs','Buddhists','Jains','Rural_Households','Urban_Households','Households','Below_Primary_Education','Primary_Education','Middle_Education','Secondary_Education','Higher_Education','Age_Group_0_29','Age_Group_30_49','Age_Group_50','Age_not_stated'
]].sum()


# # Analysis and Visualizations
# 
# Here we will use Libraries like ``` matplotlib``` and ```seaborn``` for visualization. Matplotlib have an alias as ```plt``` and seaborn is aliased as ```sns```.

# ### Q1 Which are the 10 most populated states/Uts of India?

# In[45]:



pop = census.groupby('state_name')[['Population']].sum().sort_values('Population', ascending = False).head(10)
plt.figure(figsize=(20,7))                                                     
plt.plot(pop.index, pop.Population, c='r', marker ='o')
plt.bar(pop.index, pop.Population)
plt.title('Top 10 most populated states/Uts', c='r', fontsize = 20)
plt.xlabel('States/Uts', c='r', fontsize = 12)
plt.ylabel('Population', c='r', fontsize = 12);


# ##### Comments
# * Uttar pradesh is the state with highest of 20.42 crore as per last census occured in 2011
# * Gujrat is on the 10th number of top list with a population of a approx 6.5 crores

# ### Q2 Which are the  5 least populated states?

# In[46]:


pop1 = census.groupby('state_name')[['Population']].sum().sort_values('Population', ascending = True).head(5)


# In[47]:


plt.figure(figsize=(15,5))
plt.bar(pop1.index, pop1.Population)
plt.title('Top 5 least populated states/Uts', fontsize=20, c='r')
plt.xlabel('States/Uts', fontsize = 12, c='r')
plt.ylabel('Population', fontsize = 12, c='r');


# * Among all the states and union territories in India, Lakshdweep is the least populated with less then 1 lakh population

# ### Q3 What is the Percentage of men and women in the country Population?

# In[48]:


total_pop = census.Population.sum()
mw_pop = census.Male.sum()
mw_pop1 = census.Female.sum()
men_percent = (mw_pop/total_pop)*100
women_percent = (mw_pop1/total_pop)*100


# In[49]:


print( 'Number of male in whole population is {}'.format(men_percent),'%')
print( 'Number of female in whole population is {}'.format(women_percent),'%')


# ### Q4 What is the weight  of different religions in the country?

# In[50]:


religion = ['Hindus','Muslims','Christians','Sikhs','Buddhists','Jains' ]
hindus = census.Hindus.sum()
muslims = census.Muslims.sum()
christians = census.Christians.sum()
sikhs = census.Sikhs.sum()
buddhist = census.Buddhists.sum()
jains = census.Jains.sum()
rel_pop = [hindus,muslims,christians,sikhs,buddhist,jains]
rel_pop = np.array([hindus,muslims,christians,sikhs,buddhist,jains])
myexplode=[0.2, 0.1, 0.2, 0.3, 0.4, 0.5]
mycolor=['Orange','Green','Grey','blue','red','purple']

plt.figure(figsize=(10,7))
plt.pie(rel_pop, labels = rel_pop, explode = myexplode, startangle = 90, colors = mycolor)
plt.legend(religion);


# * Highest population in the country is of Hindus and lowest is of Jains

# ### Q5 Which are the Top 10 states/Uts with highest literacy rate?

# In[51]:


literacy_rate = census.groupby('state_name')[['Population']].sum()
ltr_df = literacy_rate

literate = census.groupby('state_name')[['Literate']].sum()

ltr_df['literate'] = literate.Literate
ltr_df['literacy_rate'] = (ltr_df.literate/ltr_df.Population)*100

literacy_rate_df=ltr_df.sort_values('literacy_rate', ascending = False).head(10)
colors = ['red','blue','green','yellow','orange','purple','pink','black','violet','brown']


# In[52]:


plt.figure(figsize=(25,9))
plt.bar(literacy_rate_df.index, literacy_rate_df.literacy_rate, color=colors)
plt.title('Top 10 states/Uts with highest literacy rate', c='r', fontsize = 20)
plt.xlabel('state/Uts_name', c='r', fontsize = 12)
plt.ylabel('Literacy rate', c='r', fontsize = 12);


# * Kerela has the highest literacy rate among all the states

# ### Q6 Is there any relation between literacy rate and population rate?

# In[53]:


literacy_rate_df1 = ltr_df.sort_values('literacy_rate', ascending = False)
plt.figure(figsize=(10,7))
sns.set_style('darkgrid')
sns.scatterplot(literacy_rate_df1.Population, literacy_rate_df1.literacy_rate, color='g', s=90)
plt.title('Comparison between literacy rate and population rate', c='r', fontsize = 20)
plt.xlabel("Population", fontsize = 12)
plt.ylabel('Literacy_rate', fontsize = 12);


# * The graph denotes that almost all the place where literacy rate is high there population rate is low

# ### Q7 What is the Education level wise_distribution in Country?

# In[54]:


below_primary_ed = census_ind_df.Below_Primary_Education.sum()
primary_ed = census_ind_df.Primary_Education.sum()
middle_ed = census_ind_df.Middle_Education.sum()
secondary_ed = census_ind_df.Secondary_Education.sum()
higher_ed = census_ind_df.Higher_Education.sum()

education_df = [below_primary_ed, primary_ed, middle_ed, secondary_ed, higher_ed]
education = ['below_primary_ed', 'primary_ed', 'middle_ed', 'secondary_ed', 'higher_ed']


# In[55]:


plt.figure(figsize=(10,7))
plt.bar(education, education_df)
plt.title('Education level wise distribution in Country', c='y', fontsize = 20)
plt.xlabel('Education level',c='r',fontsize = 12)
plt.ylabel("Population", c='r',fontsize = 12);


# * Only 0.75 crores of Population is educated till higher education
# * Approx 2 crores have gained the Primary education

# ### Q8 What is the Age-wise Population distribution?

# In[56]:


age1 = census_ind_df.Age_Group_0_29.sum()
age2 = census_ind_df.Age_Group_30_49.sum()
age3 = census_ind_df.Age_Group_50.sum()
age4 = census_ind_df.Age_not_stated.sum()
Popul = [age1, age2, age3, age4]
age_group = ['0-29', '30-49', '50', 'age_not_stated']


# In[57]:


plt.figure(figsize=(10,7))
sns.barplot(Popul, age_group)
plt.title('Age-wise Population data', fontsize = 20, c='r')
plt.xlabel("Population", fontsize = 12)
plt.ylabel('Age_group', fontsize = 12);


# * Out of total population Youths have the highest popultion in India 

# ### Q9 What is the count of Scheduled Casts & Scheduled Tribes in the country?

# In[58]:


sc= census_ind_df.SC.sum()
st= census_ind_df.ST.sum()

sct =[sc, st]
sct1 =['sc', 'st']


# In[59]:


census_ind_df.SC.sum()


# In[60]:


census_ind_df.ST.sum()


# In[61]:


plt.figure(figsize=(7,7))
plt.bar(sct1, sct, color='y')
plt.title('Count of Scheduled Castes & Scheduled Tribes', fontsize=20)
plt.xlabel('Casts', fontsize = 12)
plt.ylabel('Population', fontsize = 12 );


# * The population of Scheduled Cast is double as the population of Scheduled Tribe

# # SUMMARY

# The Dataset used in the above Analysis is the census that took place in the year 2011. It consists the columns Like
# * State name
# * District name
# * Population
# * Male
# * Female
# * SC
# * ST
# * Age
# * Education etc.
# 
# The Analysis is basically divide into 3 parts:-
# 
# ###  1.) Exploring Dataset
#  In this Part the main focus is given over Exploring datasets and getting the metadata out of it.
#  ###### Observations
#  * number of rows(640)
#  * number of columns(118)
#  * Describing the various arithematic calculations
#  * Getting basic info about the dataset like its RangeIndex, Column, dtype, memoryusage
#  
#  
# ### 2.) Filtering Dataset
# In this part, the dataset is filtered and is converted into the form that can be used for the main Analysis Part
# ###### Steps Included
# * Extracting the required column from the Dataset and Exclusing the Unwanted columns
# * Changing the names of the Columns which were in the dodgy format or were incompatible for the analysis
# * Making it reliable for futher Analysis
# 
# ### 3.) Analysis and Visualization
# In this part, the main Analysis takes place in the form of some Questions and it's answer in the form of visualization
# ###### Questions.
# * Which are the 10 most populated States/Uts of India?
# * Which are the top 5 least populated States/Uts?
# * What is the Percentage of men and women in the country Population?
# * What is the weight of different religions in the country?
# * Which are the Top 10 States/Uts with highest literacy rate?
# * Is there any relation between literacy rate and population rate?
# * What is the Education level wise_distribution in Country?
# * What is the Age-wise Population distribution?
# * What is the count of Scheduled Casts & Scheduled Tribes in the country?
# 
# ###### Observations
# 1. Uttar Pradesh is the most populated State among all the States/Uts in India
# 2. Lakshdweed is the least populated union territory among all the States/Uts 
# 3. The Percentage of men in the country is 57.47% approximately and that of women is 49.52% approx
# 4. The Highest Population in the country is of Hindus and the lowest is of Jains
# 5. Kerala has the Highest literacy rate among all the States/Uts
# 6. There is relation belween literacy rate and population rate, as the literacy rate goes high the Population rate goes down
# 7. Approx 2 crores of the  population have gained the Primary education but Only 0.75 crores of Population is educated till higher education
# 8. Youths have the highest population in the country among all age groups
# 9. The population of Scheduled Tribes(STs) in the country is approx. 10.5 crores and that of Scheduled Casts(SCs) is approx. 20 crores 
# 
# 
# ### Source of Dataset
# Kaggles: https://www.kaggle.com/danofer/india-census?select=india-districts-census-2011.csv
# 
# 
# **The Data is subjected to keep changing time to time**

# In[ ]:


jovian.commit(project = 'India_Census_2011')


# In[ ]:




