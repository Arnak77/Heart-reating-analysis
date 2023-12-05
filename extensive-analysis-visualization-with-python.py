#!/usr/bin/env python
# coding: utf-8

# ## 1. Import libraries 
# 
# 

# We can see that the input folder contains one input file named `heart.csv`.

# In[113]:


import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")


# In[114]:


# ignore warnings

import warnings
warnings.filterwarnings('ignore')


# I have imported the libraries. The next step is to import the datasets.

# ## 2. Import dataset 
# I will import the dataset with the usual `pandas read_csv()` function which is used to import CSV (Comma Separated Value) files.
# 

# In[115]:


df=pd.read_csv(r"D:\NIT\2 DEC\24th- Seaborn, Eda practicle\EDA\heart.csv")


# ## 3. Exploratory Data Analysis 
# 
# 
# The scene has been set up. Now let the actual fun begin.

# #### Check shape of the dataset 
# 
# - It is a good idea to first check the shape of the dataset.

# In[116]:


# print the shape
print('The shape of the dataset : ', df.shape)


# Now, we can see that the dataset contains 303 instances and 14 variables.

# #### Preview the dataset <a class="anchor" id="6.2"></a>
# 
# 

# In[117]:


# preview dataset
df.head()


# #### Summary of dataset <a class="anchor" id="6.3"></a>

# In[118]:


# summary of dataset
df.info()


# #### Dataset description
# 
# - The dataset contains several columns which are as follows -
# 
#   - age : age in years
#   - sex : (1 = male; 0 = female)
#   - cp : chest pain type
#   - trestbps : resting blood pressure (in mm Hg on admission to the hospital)
#   - chol : serum cholestoral in mg/dl
#   - fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#   - restecg : resting electrocardiographic results
#   - thalach : maximum heart rate achieved
#   - exang : exercise induced angina (1 = yes; 0 = no)
#   - oldpeak : ST depression induced by exercise relative to rest
#   - slope : the slope of the peak exercise ST segment
#   - ca : number of major vessels (0-3) colored by flourosopy
#   - thal : 3 = normal; 6 = fixed defect; 7 = reversable defect
#   - target : 1 or 0

# #### Check the data types of columns
# 
# 
# - The above `df.info()` command gives us the number of filled values along with the data types of columns.
# 
# - If we simply want to check the data type of a particular column, we can use the following command.

# In[119]:


df.dtypes


# #### Important points about dataset 
# 
# 
# - `sex` is a character variable. Its data type should be object. But it is encoded as (1 = male; 0 = female). So, its data type is given as int64.
# 
# - Same is the case with several other variables - `fbs`, `exang` and `target`.
# 
# - `fbs (fasting blood sugar)` should be a character variable as it contains only 0 and 1 as values (1 = true; 0 = false). As it contains only 0 and 1 as values, so its data type is given as int64.
# 
# - `exang (exercise induced angina)` should also be a character variable as it contains only 0 and 1 as values (1 = yes; 0 = no). It also contains only 0 and 1 as values, so its data type is given as int64.
# 
# - `target` should also be a character variable. But, it also contains 0 and 1 as values. So, its data type is given as int64.
# 

# #### Statistical properties of dataset <a class="anchor" id="6.7"></a>

# In[120]:


# statistical properties of dataset
df.describe()


# In[121]:


df.describe(include='all')


# #### Important points to note
# 
# 
# - The above command `df.describe()` helps us to view the statistical properties of numerical variables. It excludes character variables.
# 
# - If we want to view the statistical properties of character variables, we should run the following command -
# 
#      `df.describe(include=['object'])`
#      
# - If we want to view the statistical properties of all the variables, we should run the following command -
# 
#      `df.describe(include='all')`      

# #### View column names 

# In[122]:


df.columns


# ## 4. Univariate analysis 

# ### Analysis of `target` feature variable
# 
# 
# - Our feature variable of interest is `target`.
# 
# - It refers to the presence of heart disease in the patient.
# 
# - It is integer valued as it contains two integers 0 and 1 - (0 stands for absence of heart disease and 1 for presence of heart disease).
# 
# - So, in this section, I will analyze the `target` variable. 
# 
# 

# #### Check the number of unique values in `target` variable

# In[123]:


df['target'].nunique()


# In[124]:


df['oldpeak'].nunique()


# We can see that there are 2 unique values in the `target` variable.

# #### View the unique values in `target` variable

# In[125]:


df['target'].unique()


# In[126]:


df['oldpeak'].unique()


# #### Comment 
# 
# So, the unique values are 1 and 0. (1 stands for presence of heart disease and 0 for absence of hear disease).

# #### Frequency distribution of `target` variable

# In[127]:


df['target'].value_counts()


# #### Comment
# 
# - `1` stands for presence of heart disease. So, there are 165 patients suffering from heart disease.
# 
# - Similarly, `0` stands for absence of heart disease. So, there are 138 patients who do not have any heart disease.
# 
# - We can visualize this information below.

# #### Visualize frequency distribution of `target` variable

# In[128]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(data=df,x='target')


# #### Interpretation
# 
# 
# - The above plot confirms the findings that -
# 
#    - There are 165 patients suffering from heart disease, and 
#    
#    - There are 138 patients who do not have any heart disease.

# #### Frequency distribution of `target` variable wrt `sex`

# In[129]:


df.groupby('sex')['target'].value_counts()


#  #### Comment
# 
# 
# - `sex` variable contains two integer values 1 and 0 : (1 = male; 0 = female).
# 
# - `target` variable also contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)
# 
# -  So, out of 96 females - 72 have heart disease and 24 do not have heart disease.
# 
# - Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease.
# 
# - We can visualize this information below.
# 

# We can visualize the value counts of the `sex` variable wrt `target` as follows -

# In[130]:


f,ax=plt.subplots(figsize=(8,7))
ax=sns.countplot(data=df,x='sex',hue='target')


# In[131]:


ax=sns.catplot(data=df,x='target',col='sex',kind='count',hue='target',height=4,aspect=3)


# In[132]:


ax=sns.catplot(data=df,x='target',col='sex',kind='count',hue='target',height=4)


# In[133]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(data=df,x='target',hue='sex')


# #### Interpretation
# 
# - We can see that the values of `target` variable are plotted wrt `sex` : (1 = male; 0 = female).
# 
# - `target` variable also contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)
# 
# - The above plot confirms our findings that -
# 
#     - Out of 96 females - 72 have heart disease and 24 do not have heart disease.
# 
#     - Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease.
# 

# Alternatively, we can visualize the same information as follows :

# In[134]:


ax=sns.catplot(data=df,x='target',col='sex',kind='count',hue='target',height=4)


# #### Comment
# 
# 
# - The above plot segregate the values of `target` variable and plot on two different columns labelled as (sex = 0, sex = 1).
# 
# - I think it is more convinient way of interpret the plots.

# We can plot the bars horizontally as follows :

# In[135]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(data=df,y='target',hue='sex')


# In[136]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(data=df,y='target',hue='sex',palette='RdBu_r')


# We can use a different color palette as follows :

# In[137]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(x='target',data=df)


# In[138]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(x='target',data=df,facecolor=(0.2,0,0.4,0.2),linewidth=10)


# We can use `plt.bar` keyword arguments for a different look :

# In[139]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=df, facecolor=(0, 0, 0, 0), linewidth=5, edgecolor=sns.color_palette("dark", 3))
plt.show()


# In[140]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(x='target',data=df,facecolor=(0,0,0,0.5),linewidth=10, edgecolor=sns.color_palette("dark",2))


# #### Comment
# 
# 
# - I have visualize the `target` values distribution wrt `sex`. 
# 
# - We can follow the same principles and visualize the `target` values distribution wrt `fbs (fasting blood sugar)` and `exang (exercise induced angina)`.

# In[141]:


df['fbs'].value_counts()


# In[142]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(data=df,x="target",hue='fbs')


# In[143]:


f,ax=plt.subplots(figsize=(8,6))
ax=sns.countplot(data=df,x="target",hue='exang')


# In[144]:


start.groupby('exang')['target'].value_counts()


# ### Findings of Univariate Analysis
# 
# Findings of univariate analysis are as follows:-
# 
# -	Our feature variable of interest is `target`.
# 
# -   It refers to the presence of heart disease in the patient.
# 
# -   It is integer valued as it contains two integers 0 and 1 - (0 stands for absence of heart disease and 1 for presence of heart disease).
# 
# - `1` stands for presence of heart disease. So, there are 165 patients suffering from heart disease.
# 
# - Similarly, `0` stands for absence of heart disease. So, there are 138 patients who do not have any heart disease.
# 
# - There are 165 patients suffering from heart disease, and 
#    
# - There are 138 patients who do not have any heart disease.
# 
# - Out of 96 females - 72 have heart disease and 24 do not have heart disease.
# 
# - Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease.
# 

# ## 5. Bivariate Analysis 

# ### Estimate correlation coefficients <a class="anchor" id="8.1"></a>
# 
# Our dataset is very small. So, I will compute the standard correlation coefficient (also called Pearson's r) between every pair of attributes. I will compute it using the `df.corr()` method as follows:-

# In[145]:


correlation = df.corr()


# The target variable is `target`. So, we should check how each attribute correlates with the `target` variable. We can do it as follows:-

# In[146]:


correlation['target'].sort_values()


# In[147]:


correlation['target'].sort_values(ascending=False)


# #### Interpretation of correlation coefficient
# 
# - The correlation coefficient ranges from -1 to +1. 
# 
# - When it is close to +1, this signifies that there is a strong positive correlation. So, we can see that there is no variable which has strong positive correlation with `target` variable.
# 
# - When it is clsoe to -1, it means that there is a strong negative correlation. So, we can see that there is no variable which has strong negative correlation with `target` variable.
# 
# - When it is close to 0, it means that there is no correlation. So, there is no correlation between `target` and `fbs`.
# 
# - We can see that the `cp` and `thalach` variables are mildly positively correlated with `target` variable. So, I will analyze the interaction between these features and `target` variable.
# 
# 

# ### Analysis of `target` and `cp` variable 

# #### Explore `cp` variable
# 
# 
# - `cp` stands for chest pain type.
# 
# - First, I will check number of unique values in `cp` variable.

# In[148]:


df['cp'].nunique()


# So, there are 4 unique values in `cp` variable. Hence, it is a categorical variable.

# Now, I will view its frequency distribution as follows :

# In[149]:


df['cp'].value_counts()


# #### Comment
# 
# - It can be seen that `cp` is a categorical variable and it contains 4 types of values - 0, 1, 2 and 3.

# #### Visualize the frequency distribution of `cp` variable

# In[150]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(data=df)
plt.show()


# In[151]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(data=df,x="cp",hue='cp')
plt.show()


# #### Frequency distribution of `target` variable wrt `cp`

# In[152]:


df.groupby('cp')['target'].value_counts()


# #### Comment
# 
# 
# - `cp` variable contains four integer values 0, 1, 2 and 3.
# 
# - `target` variable contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)
# 
# - So, the above analysis gives `target` variable values categorized into presence and absence of heart disease and groupby `cp` variable values.
# 
# - We can visualize this information below.

# We can visualize the value counts of the `cp` variable wrt `target` as follows -

# In[153]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp", hue="target", data=df)
plt.show()


# #### Interpretation
# 
# - We can see that the values of `target` variable are plotted wrt `cp`.
# 
# - `target` variable contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 = Absence of heart disease)
# 
# - The above plot confirms our above findings, 

# Alternatively, we can visualize the same information as follows :

# In[154]:


ax=sns.catplot(data=df,x='target',col='cp',kind='count',hue='cp',height=2)


# ### Analysis of `target` and `thalach` variable <a class="anchor" id="8.3"></a>
# 

# #### Explore `thalach` variable
# 
# 
# - `thalach` stands for maximum heart rate achieved.
# 
# - I will check number of unique values in `thalach` variable as follows :

# In[155]:


df['thalach'].nunique()


# - So, number of unique values in `thalach` variable is 91. Hence, it is numerical variable.
# 
# - I will visualize its frequency distribution of values as follows :

# #### Visualize the frequency distribution of `thalach` variable

# In[156]:


f, ax = plt.subplots(figsize=(10,6))
ax.grid(linestyle='-', alpha=0.7)
ax=sns.distplot(df['thalach'],bins=8 )


# #### Comment
# 
# - We can see that the `thalach` variable is slightly negatively skewed.

# We can use Pandas series object to get an informative axis label as follows :

# In[157]:


f, ax = plt.subplots(figsize=(10,6))
ax.grid(linestyle='-', alpha=0.7)
ax=sns.distplot(df['thalach'],bins=8 )
#start['thalach']=pd.series(start['thalach'],name="thalach variable")


# We can plot the distribution on the vertical axis as follows:-

# In[158]:


f, ax = plt.subplots(figsize=(10,6))
ax.grid(linestyle='-', alpha=0.7)
ax=sns.distplot(df['thalach'],bins=8 ,vertical=True)


# #### Seaborn Kernel Density Estimation (KDE) Plot
# 
# 
# - The kernel density estimate (KDE) plot is a useful tool for plotting the shape of a distribution.
# 
# - The KDE plot plots the density of observations on one axis with height along the other axis.
# 
# - We can plot a KDE plot as follows :

# In[159]:


f, ax = plt.subplots(figsize=(10,6))
ax = sns.kdeplot(df['thalach'])


# We can shade under the density curve and use a different color as follows:

# In[160]:


f,ax=plt.subplots(figsize=(10,6))
ax=sns.kdeplot(df['thalach'],color="r",shade=True)


# #### Histogram
# 
# - A histogram represents the distribution of data by forming bins along the range of the data and then drawing bars to show the number of observations that fall in each bin.
# 
# - We can plot a histogram as follows :

# In[161]:


f, ax = plt.subplots(figsize=(10,6))
ax=sns.distplot(df['thalach'],kde=False,rug=True)


# #### Visualize frequency distribution of `thalach` variable wrt `target`

# In[162]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df,hue="target")


# In[ ]:





# #### Interpretation
# 
# - We can see that those people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).

# We can add jitter to bring out the distribution of values as follows :

# In[163]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df,hue="target",jitter=False)


# #### Visualize distribution of `thalach` variable wrt `target` with boxplot

# In[164]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="thalach", data=df,hue="target")


# #### Interpretation
# 
# The above boxplot confirms our finding that people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).

# ### Findings of Bivariate Analysis 
# 
# Findings of Bivariate Analysis are as follows –
# 
# 
# - There is no variable which has strong positive correlation with `target` variable.
# 
# - There is no variable which has strong negative correlation with `target` variable.
# 
# - There is no correlation between `target` and `fbs`.
# 
# - The `cp` and `thalach` variables are mildly positively correlated with `target` variable. 
# 
# - We can see that the `thalach` variable is slightly negatively skewed.
# 
# - The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).
# 
# - The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).
# 

# ## 6. Multivariate analysis
# 
# 
# 
# - The objective of the multivariate analysis is to discover patterns and relationships in the dataset.

# ### Discover patterns and relationships
# 
# - An important step in EDA is to discover patterns and relationships between variables in the dataset. 
# 
# - I will use `heat map` and `pair plot` to discover the patterns and relationships in the dataset.
# 
# - First of all, I will draw a `heat map`.

# ### Heat Map 

# In[165]:


plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Heart Disease Dataset')
h=sns.heatmap(df.corr(),annot=True)


# In[166]:


plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Heart Disease Dataset')
h=sns.heatmap(df.corr(),annot=True,fmt='.2f',linecolor='red')


# #### Interpretation
# 
# From the above correlation heat map, we can conclude that :-
# 
# - `target` and `cp` variable are mildly positively correlated (correlation coefficient = 0.43).
# 
# - `target` and `thalach` variable are also mildly positively correlated (correlation coefficient = 0.42).
# 
# - `target` and `slope` variable are weakly positively correlated (correlation coefficient = 0.35).
# 
# - `target` and `exang` variable are mildly negatively correlated (correlation coefficient = -0.44).
# 
# - `target` and `oldpeak` variable are also mildly negatively correlated (correlation coefficient = -0.43).
# 
# - `target` and `ca` variable are weakly negatively correlated (correlation coefficient = -0.39).
# 
# - `target` and `thal` variable are also waekly negatively correlated (correlation coefficient = -0.34).
# 
# 
# 

# ### Pair Plot 

# In[167]:


sns.pairplot(data=start)


# In[168]:


num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target' ]
sns.pairplot(df[num_var], kind='scatter', diag_kind='hist')
plt.show()


# #### Comment
# 
# 
# - I have defined a variable `num_var`. Here `age`, `trestbps`, ``chol`, `thalach` and `oldpeak`` are numerical variables and `target` is the categorical variable.
# 
# - So, I wll check relationships between these variables.

# ### Analysis of `age` and other variables

# #### Check the number of unique values in `age` variable

# In[169]:


df['age'].nunique()


# #### View statistical summary of `age` variable

# In[170]:


df['age'].describe()


# #### Interpretation
# 
# - The mean value of the `age` variable is 54.37 years.
# 
# - The minimum and maximum values of `age` are 29 and 77 years.

# #### Plot the distribution of `age` variable
# 
# Now, I will plot the distribution of `age` variable to view the statistical properties.

# In[171]:


f, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(df['age'], bins=10)


# #### Interpretation
# 
# - The `age` variable distribution is approximately normal.

# ### Analyze `age` and `target` variable

# #### Visualize frequency distribution of `age` variable wrt `target`

# In[172]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="age", data=df,hue='target')


# #### Interpretation
# 
# - We can see that the people suffering from heart disease (target = 1) and people who are not suffering from heart disease (target = 0) have comparable ages.

# #### Visualize distribution of `age` variable wrt `target` with boxplot

# In[173]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="age", data=df,hue='target')


# #### Interpretation
# 
# - The above boxplot tells two different things :
# 
#   - The mean age of the people who have heart disease is less than the mean age of the people who do not have heart disease.
#   
#   - The dispersion or spread of age of the people who have heart disease is greater than the dispersion or spread of age of the people who do not have heart disease.
# 

# ### Analyze `age` and `trestbps` variable
# 
# 

# I will plot a scatterplot to visualize the relationship between `age` and `trestbps` variable.

# In[174]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(x="age", data=df, y="trestbps")


# #### Interpretation
# 
# - The above scatter plot shows that there is no correlation between `age` and `trestbps` variable.

# In[175]:


ax = sns.regplot(x="age", y="trestbps", data=df)


# #### Interpretation
# 
# - The above line shows that linear regression model is not good fit to the data.

# ### Analyze `age` and `chol` variable

# In[176]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(x="age", y="chol", data=df)


# In[177]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="age", y="chol", data=df)


# #### Interpretation
# 
# - The above plot confirms that there is a slighly positive correlation between `age` and `chol` variables.

# ### Analyze `chol` and `thalach` variable

# In[178]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(x="chol", y = "thalach", data=df)


# In[179]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="chol", y="thalach", data=df)


# #### Interpretation
# 
# 
# - The above plot shows that there is no correlation between `chol` and `thalach` variable.

# ## 7. Dealing with missing values 
# 
# 
# 
# 
# 	In Pandas missing data is represented by two values:
# 
#    None is a Python singleton object that is often used for missing data in Python code.
#   
#  -NaN (an acronym for Not a Number), is a special floating-point value recognized by all systems that use the standard IEEE floating-point representation.
# 
# 
# -  There are different methods in place on how to detect missing values.
# 
# 
# ### Pandas isnull() and notnull() functions 
# 
# 
# - Pandas offers two functions to test for missing data - `isnull()` and `notnull()`. These are simple functions that return a boolean value indicating whether the passed in argument value is in fact missing data.
# 
# -  Below, I will list some useful commands to deal with missing values.
# 
# 
# ### Useful commands to detect missing values 
# 
# -	**df.isnull()**
# 
# The above command checks whether each cell in a dataframe contains missing values or not. If the cell contains missing value, it returns True otherwise it returns False.
# 
# 
# -	**df.isnull().sum()**
# 
# The above command returns total number of missing values in each column in the dataframe.
# 
# 
# -	**df.isnull().sum().sum()** 
# 
# It returns total number of missing values in the dataframe.
# 
# 
# -	**df.isnull().mean()**
# 
# It returns percentage of missing values in each column in the dataframe.
# 
# 
# -	**df.isnull().any()**
# 
# It checks which column has null values and which has not. The columns which has null values returns TRUE and FALSE otherwise.
# 
# -	**df.isnull().any().any()**
# 
# It returns a boolean value indicating whether the dataframe has missing values or not. If dataframe contains missing values it returns TRUE and FALSE otherwise.
# 
# 
# -	**df.isnull().values.any()**
# 
# It checks whether a particular column has missing values or not. If the column contains missing values, then it returns TRUE otherwise FALSE.
# 
# 
# -	**df.isnull().values.sum()**
# 
# 
# It returns the total number of missing values in the dataframe.
# 
# 

# In[180]:


df.isnull()


# In[200]:


# check for missing values

df.isnull().sum()


# In[182]:


df.isnull().sum().sum()


# In[183]:


df.isnull().mean()


# In[184]:


df.isnull().any()


# In[185]:


df.isnull().any().any()


# In[186]:


df.isnull().values.any()


# In[187]:


df.isnull().values.sum()


# #### Interpretation
# 
# We can see that there are no missing values in the dataset.

# ## 8. Check with ASSERT statement 
# 
# - We must confirm that our dataset has no missing values. 
# 
# - We can write an **assert statement** to verify this. 
# 
# - We can use an assert statement to programmatically check that no missing, unexpected 0 or negative values are present. 
# 
# - This gives us confidence that our code is running properly.
# 
# - **Assert statement** will return nothing if the value being tested is true and will throw an AssertionError if the value is false.
# 
# - **Asserts**
# 
#   - assert 1 == 1 (return Nothing if the value is True)
# 
#   - assert 1 == 2 (return AssertionError if the value is False)

# In[204]:


#assert that there are no missing values in the dataframe

pd.notnull(df).all().all()


# In[205]:


#assert all values are greater than or equal to 0

(df >= 0).all().all()


# #### Interpretation
# 
# - The above two commands do not throw any error. Hence, it is confirmed that there are no missing or negative values in the dataset. 
# 
# - All the values are greater than or equal to zero.

# ## 9. Outlier detection 

# I will make boxplots to visualise outliers in the continuous numerical variables : -
# 
# `age`, `trestbps`, `chol`, `thalach` and  `oldpeak` variables.
# 

# ### `age` variable

# In[190]:


df['age'].describe()


# #### Box-plot of `age` variable

# In[191]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(df,x="age")


# ### `trestbps` variable

# In[192]:


df['trestbps'].describe()


# #### Box-plot of `trestbps` variable

# In[193]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["trestbps"])


# ### `chol` variable

# In[194]:


df['chol'].describe()


# #### Box-plot of `chol` variable

# In[195]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["chol"])


# ### `thalach` variable

# In[196]:


df['thalach'].describe()


# #### Box-plot of `thalach` variable

# In[197]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["thalach"])
plt.show()


# ### `oldpeak` variable

# In[198]:


df['oldpeak'].describe()


# #### Box-plot of `oldpeak` variable

# In[199]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["oldpeak"])


# #### Findings
# 
# - The `age` variable does not contain any outlier.
# 
# - `trestbps` variable contains outliers to the right side.
# 
# - `chol` variable also contains outliers to the right side.
# 
# - `thalach` variable contains a single outlier to the left side.
# 
# - `oldpeak` variable contains outliers to the right side.
# 
# - Those variables containing outliers needs further investigation.
# 

# In[ ]:




