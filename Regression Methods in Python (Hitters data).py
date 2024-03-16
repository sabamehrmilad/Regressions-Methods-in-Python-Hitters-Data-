#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# In[2]:


data1= pd.read_csv('D:\\Data science\\S29\\CS_05.csv')
data1


# In[3]:


data1.head()


# In[4]:


data1.shape


# In[5]:


data1.info()


# In[6]:


data1.isna().sum()


# In[7]:


data1.info()


# In[8]:


#dealing with w/MVws
#Analysis of MVs should be done 
#remove records with MVs 
data2=data1.dropna(subset= ['Salary'], inplace=False)
data2


# In[9]:


data2.isna().sum()


# In[10]:


data2 = data2.iloc[:, 1:]


# In[11]:


data2


# In[12]:


data2.describe()


# In[13]:


data2.info()


# In[14]:


#Continuous variables distribution
var_ind = list(range(13)) + list(range(15, 19))
plot = plt.figure(figsize = (12, 6))
plot.subplots_adjust(hspace = 0.5, wspace = 0.5)
for i in range(1, 18):
    a = plot.add_subplot(3, 6, i)
    a.hist(data2.iloc[: , var_ind[i - 1]], alpha = 0.7)
    a.title.set_text(data2.columns[var_ind[i - 1]])


# In[15]:


#Box plo of price 
plt.boxplot(data2['Salary'], showmeans=True )
plt.title('Boxplot of salary ')


# In[16]:


var_ind


# In[17]:


corr_table=round(data2.iloc[:, var_ind].corr(method='pearson'), 2)


# In[18]:


corr_table


# In[19]:


plt.figure(figsize=(12,6))
sns.heatmap(corr_table, annot= True)


# In[20]:


#Scatter Plot
var_ind = list(range(13)) + list(range(15, 18))
plot = plt.figure(figsize = (12, 12))
plot.subplots_adjust(hspace = 0.8, wspace = 0.5)
for i in range(1, 17):
    a = plot.add_subplot(4, 4, i)
    a.scatter(x = data2.iloc[: , var_ind[i - 1]], y = data2.iloc[: , 18], alpha = 0.5)
    a.title.set_text('Salary vs. ' + data2.columns[var_ind[i - 1]])


# In[21]:


data2['League'].value_counts()


# In[22]:


data2['Division'].value_counts()


# In[23]:


data2['NewLeague'].value_counts()


# # Data Prepration 

# Divide dataset into train and test 

# In[24]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data2, test_size = 0.2, random_state = 1234)


# In[25]:


train.head()


# In[26]:


test.head()


# In[27]:


train.shape


# In[28]:


test.shape


# In[29]:


train.describe()


# In[30]:


test.describe()


# # Building Prediction Model 

# # Model1: Linear Regression 

# In[31]:


#Creat Dummies for columns with categorical variables 
dummies= pd.get_dummies(train[['League','Division','NewLeague']])
dummies.head()


# In[32]:


#Define the feature set X 
X_ = train.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1)
X_train = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train = sm.add_constant(X_train) #adding a constant

#Define response variable
y_train = train['Salary']


# In[33]:


X_train.head()


# In[34]:


#Regression Model 
lm= sm.OLS(y_train,X_train).fit()
lm.summary()


# In[35]:


#Check Assumptions of Regression
#Normality of residuals

#Plot histogram of residuals
sns.histplot(lm.resid, stat='probability', kde=True, alpha=0.7, color='green',  bins = np.linspace(min(lm.resid), max(lm.resid), 20))


# In[36]:


#QQ-plot 
qqplot_lm= sm.qqplot(lm.resid, line='s')
plt.show()


# In[37]:


#Test for Skewness and Kurtosis
#Good for sample size > 25

#Jarque-Bera Test (Skewness = 0 ?)
#H0: the data is normally distributed
#p-value < 0.05 reject normality assumption

#Omnibus K-squared normality test
#The Omnibus test combines the random variables for Skewness and Kurtosis into a single test statistic
#H0: the data is normally distributed
#p-value < 0.05 reject normality assumption

lm.summary()


# In[38]:


#Residuals vs. Fitted Values
sns.regplot(x = lm.fittedvalues, y = lm.resid, lowess = True, 
                       scatter_kws = {"color": "black"}, line_kws = {"color": "red"})
plt.xlabel('Fitted Values', fontsize = 12)
plt.ylabel('Residuals', fontsize = 12)
plt.title('Residuals vs. Fitted Values', fontsize = 12)
plt.grid()


# In[39]:


#Check Cook's distance 
sum(lm.get_influence().summary_frame().cooks_d >1)


# In[40]:


X_train.info()


# In[41]:


#Check Multicollinearity
#Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)

calc_vif(X_train.iloc[:, 1:])
#If VIF > 10 then multicollinearity is high


# In[42]:


lm.summary()


# In[43]:


#Regression Model based on t-test results
lm = sm.OLS(y_train, X_train[['const', 'AtBat', 'Hits','Walks','PutOuts', 'Division_W']]).fit()
lm.summary()


# # Model1: Test the model 

# In[44]:


#Create dummies for columns with categorical variables
dummies = pd.get_dummies(test[['League', 'Division', 'NewLeague']])

#Define the feature set X 
X_ = test.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1)
X_test = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_test = sm.add_constant(X_test) # adding a constant

#Define response variable
y_test = test['Salary']


# In[45]:


X_test.head()


# In[46]:


y_test.head()


# In[47]:


pred_lm= lm.predict(X_test[['const', 'AtBat', 'Hits','Walks','PutOuts', 'Division_W']])


# In[48]:


pred_lm


# In[49]:


#Absolute Error 
abs_err_lm= abs(y_test-pred_lm)


# In[50]:


#Absolute error mean, median, sd, IQR, max, min
from scipy.stats import iqr
model_comp = pd.DataFrame({'Mean of AbsErrors':    abs_err_lm.mean(),
                           'Median of AbsErrors' : abs_err_lm.median(),
                           'SD of AbsErrors' :     abs_err_lm.std(),
                           'IQR of AbsErrors':     iqr(abs_err_lm),
                           'Min of AbsErrors':     abs_err_lm.min(),
                           'Max of AbsErrors':     abs_err_lm.max()}, index = ['LM_t-test'])
model_comp


# In[51]:


pred_lm = lm.predict(X_test[['const', 'AtBat', 'Hits','Walks','PutOuts', 'Division_W']])


# In[52]:


abs_err_lm = abs(y_test - pred_lm)


# In[53]:


#Absolute error mean, median, sd, IQR, max, min
from scipy.stats import iqr
model_comp = pd.DataFrame({'Mean of AbsErrors':    abs_err_lm.mean(),
                           'Median of AbsErrors' : abs_err_lm.median(),
                           'SD of AbsErrors' :     abs_err_lm.std(),
                           'IQR of AbsErrors':     iqr(abs_err_lm),
                           'Min of AbsErrors':     abs_err_lm.min(),
                           'Max of AbsErrors':     abs_err_lm.max()}, index = ['LM_t-test'])
model_comp


# In[54]:


plt.scatter(x = y_test, y = pred_lm)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')


# # BoxCox Transformation 

# In[55]:


#Box-Cox Transformation 
from scipy.stats import boxcox
box_results= boxcox(y_train, alpha=0.05)


# In[56]:


box_results


# In[57]:


#Log transformation 
logy_train=np.log(y_train)
logy_train


# In[58]:


#Histogram of salary 
sns.histplot(y_train, stat = 'probability', 
             kde = True, alpha = 0.7, color = 'green',
             bins = np.linspace(min(y_train), max(y_train), 20))


# In[59]:


#qq-plot 
qqplot_lm_bc = sm.qqplot(y_train, line = 's')
plt.show()


# In[60]:


#Histogram of Log Salary
sns.histplot(logy_train, stat = 'probability', 
             kde = True, alpha = 0.7, color = 'green',
             bins = np.linspace(min(logy_train), max(logy_train), 20))


# In[61]:


#QQ-plot
qqplot_lm_bc = sm.qqplot(logy_train, line = 's')
plt.show()


# # Model 2: Linear Regression Using the Best Subset selection 

# In[62]:


X_train.head()


# In[63]:


X_train


# In[73]:


#Define function to fit linear regression 
def fit_lm(feature_set):
    reg_model= sm.OLS(logy_train, X_train[['const'] + list(feature_set)]).fit()
    return{'model' : reg_model, 'RSquared' : reg_model.rsquared}
 


# In[74]:


def bestsubset_func(k):
    
    res= []
    
    #Looping over all possible combination 
    for features in itertools.combinations(X_train.iloc[:, 1:].columns, k):
        res.append(fit_lm(features))
        
    models= pd.DataFrame(res)
    
    #Choose the model with Highest RSquared 
    best_model = models.iloc[models['RSquared'].argmax()]
    
    #Return the best model 
    return best_model 


# In[66]:


['const', 'Hits', 'CHits', 'Division_W']


# In[75]:


import time # to measure the processing time 
models_bestsub = pd.DataFrame(columns = ['RSquared', 'model'])

start_time = time.time()
for i in range(1, len(X_train.iloc[:, 1:].columns) +1):
    models_bestsub.loc[i] = bestsubset_func(i)
end_time = time.time()
print('The Processing time is: ', end_time - start_time, 'seconds')

    


# In[68]:


models_bestsub


# In[69]:


models_bestsub.loc[4, 'model'].summary()


# In[70]:


models_bestsub_adjrs = models_bestsub.apply(lambda row:row[1].rsquared_adj, axis=1)
models_bestsub_adjrs


# In[ ]:


models_bestsub_adjrs.max()


# In[ ]:


#Adj. RSquared Plot 
plt.plot(models_bestsub_adjrs)
plt.xlabel('Predictors')
plt.xticks(range(1,20))
plt.ylabel('Adj RSquared')
plt.axvline(models_bestsub_adjrs.argmax() + 1, color='red', linewidth= 2, linestyle='--')


# In[ ]:


models_bestsub_aic = models_bestsub.apply(lambda row:row[1].aic, axis=1)
plt.plot(models_bestsub_aic)
plt.xlabel('Predictors')
plt.xticks(range(1,20))
plt.ylabel('AIC')
plt.axvline(models_bestsub_aic.argmin() + 1, color='red', linewidth= 2, linestyle='--')


# In[ ]:


models_bestsub.loc[11, 'model'].model.exog_names


# # Model2: Prediction on Test Dataset

# In[ ]:


pred_bestsub= models_bestsub.loc[11, 'model'].predict(X_test[models_bestsub.loc[11, 'model'].model.exog_names])


# In[ ]:


pred_bestsub = np.exp(pred_bestsub)
pred_bestsub.head()


# In[ ]:


y_test


# In[ ]:


pred_bestsub = np.exp(pred_bestsub)
pred_bestsub.head()


# In[ ]:


abs_err_bestsub = abs(y_test - pred_bestsub)


# In[ ]:


from scipy.stats import iqr
model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_bestsub.mean(),
                                             'Median of AbsErrors' : abs_err_bestsub.median(),
                                             'SD of AbsErrors' :     abs_err_bestsub.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_bestsub),
                                             'Min of AbsErrors':     abs_err_bestsub.min(),
                                             'Max of AbsErrors':     abs_err_bestsub.max()}, index = ['BestSubset']), 
                               ignore_index = False)

model_comp


# In[ ]:


# Actual vs Prediction 
plt.scatter( x= y_test, y= pred_bestsub)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs Prediction')

#Add 45 degree line 
xp = np.linspace(y_test.min() , y_test.max() , 100)
plt.plot(xp, xp, alpha = 0.9, linewidth= 2, color= 'red')


# # Model 3: Forward and Backward Stepwise Selection
# 

# In[ ]:


def forward_func(features):
    res=[]
    #pull out features still needed to process 
    remaining_features = [_ for _ in X_train.iloc[: , 1:].columns if _ not in features]
    
    for f in remaining_features:
        res.append(fit_lm(features + [f]))
        
    models=pd.DataFrame(res)    
    
    #finding the best model 
    best_model = models.iloc[models['RSquared'].argmax()]
    
    return best_model 


# In[ ]:


#Forward Selection Implementation
models_fw = pd.DataFrame(columns = ['RSquared', 'model'])
features = []
for i in range(1, len(X_train.iloc[:, 1:].columns) + 1):
    models_fw.loc[i] = forward_func(features)
    features = models_fw.loc[i, 'model'].model.exog_names[1:]
    


# In[ ]:


features


# In[ ]:


models_fw[:].max()


# In[ ]:


models_fw.loc[4, 'model'].summary()


# # Doing Prediction 

# In[ ]:


models_fw.loc[12, 'model'].summary()


# In[ ]:


pred_fw = models_fw.loc[12, 'model'].predict(X_test[models_fw.loc[12, 'model'].model.exog_names])
pred_fw


# In[ ]:


pred_fw = np.exp(pred_fw)
pred_fw


# In[ ]:


#Absolute Error 
abs_err_fw = abs(y_test - pred_fw)


# In[ ]:


from scipy.stats import iqr
model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_fw.mean(),
                                             'Median of AbsErrors' : abs_err_fw.median(),
                                             'SD of AbsErrors' :     abs_err_fw.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_fw),
                                             'Min of AbsErrors':     abs_err_fw.min(),
                                             'Max of AbsErrors':     abs_err_fw.max()}, index = ['Forward Stepwise']), 
                               ignore_index = False)

model_comp


# In[ ]:


#Actual VS Prediction 
plt.scatter(x=y_test, y=pred_fw)
plt.xlabel('prediction')
plt.ylabel('Actual')
plt.title('Actual VS Prediction')
# Add 45 degree Line 
xp = np.linspace(y_test.min() , y_test.max() , 100)
plt.plot(xp, xp, alpha=0.9, linewidth= 2, color= 'red')


# # Model 5: Ridge Regression 

# In[76]:


from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha = 0.1, normalize = True)
ridge_reg = ridgereg.fit(X_train, logy_train)


# In[77]:


ridge_reg.coef_


# In[83]:


lambda_grid = 10** np.linspace(-3,3,100)
lambda_grid


# In[84]:


#K-fold Cross Validation to Choose the Best Model
from sklearn.model_selection import cross_val_score

cv_errors = np.zeros(shape = len(lambda_grid)) #to save cv results

for i in range(len(lambda_grid)):
    ridgereg = Ridge(alpha = lambda_grid[i], normalize = True)
    scores = cross_val_score(estimator = ridgereg, 
                             X = X_train, y = logy_train,
                             scoring = 'neg_root_mean_squared_error',
                             cv = 5, n_jobs = -1)
    cv_errors [i] = scores.mean() 
#To check scoring: https://scikit-learn.org/stable/modules/model_evaluation.html


# In[85]:


cv_errors


# In[86]:


np.max(cv_errors)


# In[87]:


np.argmax(cv_errors)


# In[88]:


best_lambda = lambda_grid[np.argmax(cv_errors)]
best_lambda

