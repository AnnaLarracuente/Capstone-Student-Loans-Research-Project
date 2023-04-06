#!/usr/bin/env python
# coding: utf-8

# # A Naive Bayes Classifier for Predicting Probability of Americans to Purchase a Home Based on Generation & Having Student Loans (2023)
# 
# ##Anna Larracuente

# In[1]:


pip install tabulate


# In[2]:


#Import Necessary Libraries & Tools

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from tabulate import tabulate


# In[3]:


#Load, Convert to Data Frame, and View First 5 Records of Student Loans Data Data Set

sl = pd.read_csv("STUDENTLOANSDS.csv")
sldf = pd.DataFrame(sl)
sldf.head()


# In[4]:


#Pop off Submission Date, Sex, Age, Gender, Marital Status Columns (SUB_DATE)

sldf.pop("SUB_DATE")
sldf.pop("SEX")
sldf.pop("AGE")
sldf.pop("GEN")
sldf.pop("MAR_STAT")
sldf.pop("IND_ANN_INC ($)")
sldf.pop("HSD_GED?")
sldf.pop("BACH_DEG?")
sldf.pop("HIGHER_ED?")
sldf.pop("CURR_STUDENT?")
sldf.pop("OWN_CAR?")
sldf.pop("HOUSING")
sldf.pop("PETS?")
sldf.pop("VACATION?")
sldf.pop("TRAV_STATE?")
sldf.pop("TRAV_STATE_Q")
sldf.pop("TRAV_US?")
sldf.pop("TRAV_US_Q")
sldf.pop("MULTI_GEN?")
sldf.pop("CHILDREN?")
sldf.pop("MOV?")
sldf.pop("HOB?")
sldf.pop("AVG_MON_HOB")
sldf.pop("DINEOUT?")
sldf.pop("AVG_WEEKLY_DINEOUT")
sldf.pop("ORDERIN?")
sldf.pop("AVG_WEEKLY_ORDERIN")
sldf.pop("DONATIONS?")
sldf.pop("ANN_EVENTS_ATT_Q")
sldf.pop("ANN_EVENTS_HOST_Q")
sldf.pop("MON_ESS_EXP ($)")
sldf.pop("ANN_NONESS_EXP ($)")
sldf.pop("LUX_Q")
sldf.pop("STUDENTLOANS?")
sldf.pop("TOT_UG_SLD ($)")
sldf.pop("TOT_G_SLD ($)")


# In[5]:


#Print the Target Variable: HOUSING_BIN

print(sldf["HOUSING_BIN"])


# In[6]:


#Split Student Loan Data Set into Training & Testing Sets: 80% Training and 20% Test

x_train, x_test, y_train, y_test = train_test_split(sldf, sldf["HOUSING_BIN"], test_size = 0.20, random_state = 26)


# In[7]:


#Create a Gaussian Naive Bayes Classifier

NBC = GaussianNB()


# In[8]:


#Train the Model using the Training Set

NBC.fit(x_train, y_train)


# In[9]:


#Predict the Response for Test Data Set

y_pred = NBC.predict(x_test)


# In[10]:


#Check Model Accuracy

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[11]:


#Create a Frequency Table of American Homeowners by Predicted Generation & Having Student Loan Debt

freqtab = pd.crosstab(index = sldf["HOUSING_BIN"], 
                      columns = [sldf["GEN_CLASS"],
                                 sldf["STUDENTLOANS_BIN"]],
                              margins = True)

freqtab


# In[12]:


#Hypothesis & Prior Probability of Homeownership by Having Student Loans

hypos = 'Own a Home', 'Do Not Own a Home'

prob = 0.40, 0.60


# In[13]:


#Prior Probabilities Summary by Gender

prior = pd.Series(prob, hypos)

print(prior)


# In[14]:


#Likelihoods by Generation (Taken From Frequency Table Above, with Laplacian Correction)

likelihoodsPostWarWithLoans = 1/1, 1/1
likelihoodsPostWarNoLoans = 2/2, 1/2

likelihoodsBoomersIWithLoans = 1/3, 1/3
likelihoodsBoomersINoLoans = 3/3, 2/3

likelihoodsBoomersIIWithLoans = 3/5, 2/5
likelihoodsBoomersIINoLoans = 14/19, 6/19

likelihoodsGenXWithLoans = 9/28, 20/28
likelihoodsGenXNoLoans = 19/28, 10/28

likelihoodsMillennialsWithLoans = 2/17, 2/17
likelihoodsMillennialsNoLoans = 2/17, 2/17

likelihoodsGenZWithLoans = 2/5, 4/5
likelihoodsGenZNoLoans = 1/1, 1/1

likelihoodsPostWarOwnersWithLoans = 1/1
likelihoodsPostWarOwnersNoLoans = 2/2
likelihoodsPostWarNonOwnersWithLoans = 1/1
likelihoodsPostWarNonOwnersNoLoans = 1/2

likelihoodsBoomersIOwnersWithLoans = 1/3
likelihoodsBoomersIOwnersNoLoans = 3/3
likelihoodsBoomersINonOwnersWithLoans = 1/3
likelihoodsBoomersINonOwnersNoLoans = 2/3

likelihoodsBoomersIIOwnersWithLoans = 3/5
likelihoodsBoomersIIOwnersNoLoans = 14/19
likelihoodsBoomersIINonOwnersWithLoans = 2/5
likelihoodsBoomersIINonOwnersNoLoans = 6/19

likelihoodsGenXOwnersWithLoans = 9/28
likelihoodsGenXOwnersNoLoans = 19/28
likelihoodsGenXNonOwnersWithLoans = 20/28
likelihoodsGenXNonOwnersNoLoans = 10/28

likelihoodsMillennialsOwnersWithLoans = 2/17
likelihoodsMillennialsOwnersNoLoans = 2/17
likelihoodsMillennialsNonOwnersWithLoans = 2/17
likelihoodsMillennialsNonOwnersNoLoans = 2/17

likelihoodsGenZOwnersWithLoans = 2/5
likelihoodsGenZOwnersNoLoans = 1/1
likelihoodsGenZNonOwnersWithLoans = 4/5
likelihoodsGenZNonOwnersNoLoans = 1/1


# In[15]:


#Likelihoods Table Summary By Generation and Ownership

data = [["PostWar", "Yes", "Owner", likelihoodsPostWarOwnersWithLoans], 
        ["PostWar", "No", "Owner", likelihoodsPostWarOwnersNoLoans],
        ["PostWar", "Yes", "Non-Owner", likelihoodsPostWarNonOwnersWithLoans],
        ["PostWar", "No", "Non-Owner", likelihoodsPostWarNonOwnersNoLoans],
        ["BoomersI", "Yes", "Owner", likelihoodsBoomersIOwnersWithLoans], 
        ["BoomersI", "No", "Owner", likelihoodsBoomersIOwnersNoLoans],
        ["BoomersI", "Yes", "Non-Owner", likelihoodsBoomersINonOwnersWithLoans],
        ["BoomersI", "No", "Non-Owner", likelihoodsBoomersINonOwnersNoLoans],
        ["BoomersII", "Yes", "Owner", likelihoodsBoomersIIOwnersWithLoans], 
        ["BoomersII", "No", "Owner", likelihoodsBoomersIIOwnersNoLoans],
        ["BoomersII", "Yes", "Non-Owner", likelihoodsBoomersIINonOwnersWithLoans],
        ["BoomersII", "No", "Non-Owner", likelihoodsBoomersIINonOwnersNoLoans],
        ["GenX", "Yes", "Owner", likelihoodsGenXOwnersWithLoans], 
        ["GenX", "No", "Owner", likelihoodsGenXOwnersNoLoans],
        ["GenX", "Yes", "Non-Owner", likelihoodsGenXNonOwnersWithLoans],
        ["GenX", "No", "Non-Owner", likelihoodsGenXNonOwnersNoLoans],
        ["Millennials", "Yes", "Owner", likelihoodsMillennialsOwnersWithLoans], 
        ["Millennials", "No", "Owner", likelihoodsMillennialsOwnersNoLoans],
        ["Millennials", "Yes", "Non-Owner", likelihoodsMillennialsNonOwnersWithLoans],
        ["Millennials", "No", "Non-Owner", likelihoodsMillennialsNonOwnersNoLoans],
        ["GenZ", "Yes", "Owner", likelihoodsGenZOwnersWithLoans], 
        ["GenZ", "No", "Owner", likelihoodsGenZOwnersNoLoans],
        ["GenZ", "Yes", "Non-Owner", likelihoodsGenZNonOwnersWithLoans],
        ["GenZ", "No", "Non-Owner", likelihoodsGenZNonOwnersNoLoans]]

col_names = ["Generation", "Student Loans?", "Home Ownership", "Likelihood"]

print(tabulate(data, headers = col_names, tablefmt = "fancy_grid"))


# In[16]:


#Unnormalized Posterior Probability Calculation for Post War Home Owners & Non-Owners With Student Loans

unnorm1 = prior * likelihoodsPostWarWithLoans

print(unnorm1)


# In[17]:


#Unnormalized Posterior Probability Calculation for Post War Home Owners & Non-Owners With No Student Loans

unnorm2 = prior * likelihoodsPostWarNoLoans

print(unnorm2)


# In[18]:


#Unnormalized Posterior Probability Calculation for Boomers I Home Owners & Non-Owners With Student Loans

unnorm3 = prior * likelihoodsBoomersIWithLoans

print(unnorm3)


# In[19]:


#Unnormalized Posterior Probability Calculation for Boomers I Home Owners & Non-Owners With No Student Loans

unnorm4 = prior * likelihoodsBoomersINoLoans

print(unnorm4)


# In[20]:


#Unnormalized Posterior Probability Calculation for Boomers II Home Owners & Non-Owners With Student Loans

unnorm5 = prior * likelihoodsBoomersIIWithLoans

print(unnorm5)


# In[21]:


#Unnormalized Posterior Probability Calculation for Boomers II Home Owners & Non-Owners With No Student Loans

unnorm6 = prior * likelihoodsBoomersIINoLoans

print(unnorm6)


# In[22]:


#Unnormalized Posterior Probability Calculation for Gen X Home Owners & Non-Owners With Student Loans

unnorm7 = prior * likelihoodsGenXWithLoans

print(unnorm7)


# In[23]:


#Unnormalized Posterior Probability Calculation for Gen X Home Owners & Non-Owners With No Student Loans

unnorm8 = prior * likelihoodsGenXNoLoans

print(unnorm8)


# In[24]:


#Unnormalized Posterior Probability Calculation for Millennials Home Owners & Non-Owners With Student Loans

unnorm9 = prior * likelihoodsMillennialsWithLoans

print(unnorm9)


# In[25]:


#Unnormalized Posterior Probability Calculation for Millennials Home Owners & Non-Owners With No Student Loans

unnorm10 = prior * likelihoodsMillennialsNoLoans

print(unnorm10)


# In[26]:


#Unnormalized Posterior Probability Calculation for Gen Z Home Owners & Non-Owners With Student Loans

unnorm11 = prior * likelihoodsGenZWithLoans

print(unnorm11)


# In[27]:


#Unnormalized Posterior Probability Calculation for Gen Z Home Owners & Non-Owners With No Student Loans

unnorm12 = prior * likelihoodsGenZNoLoans

print(unnorm12)


# In[28]:


#Summation of Unnormalized Posterior Probability Calculation for Post War Home Owners & Non-Owners With Student Loans

prob_data1 = unnorm1.sum()

print(prob_data1)


# In[29]:


#Summation of Unnormalized Posterior Probability Calculation for Post War Home Owners & Non-Owners With No Student Loans

prob_data2 = unnorm2.sum()

print(prob_data2)


# In[30]:


#Summation of Unnormalized Posterior Probability Calculation for Boomers I Home Owners & Non-Owners With Student Loans

prob_data3 = unnorm3.sum()

print(prob_data3)


# In[31]:


#Summation of Unnormalized Posterior Probability Calculation for Boomers I Home Owners & Non-Owners With No Student Loans

prob_data4 = unnorm4.sum()

print(prob_data4)


# In[32]:


#Summation of Unnormalized Posterior Probability Calculation for Boomers II Home Owners & Non-Owners With Student Loans

prob_data5 = unnorm5.sum()

print(prob_data5)


# In[33]:


#Summation of Unnormalized Posterior Probability Calculation for Boomers II Home Owners & Non-Owners With No Student Loans

prob_data6 = unnorm6.sum()

print(prob_data6)


# In[34]:


#Summation of Unnormalized Posterior Probability Calculation for Gen X Home Owners & Non-Owners With Student Loans

prob_data7 = unnorm7.sum()

print(prob_data7)


# In[35]:


#Summation of Unnormalized Posterior Probability Calculation for Gen X Home Owners & Non-Owners With No Student Loans

prob_data8 = unnorm8.sum()

print(prob_data8)


# In[36]:


#Summation of Unnormalized Posterior Probability Calculation for Millennials Home Owners & Non-Owners With Student Loans

prob_data9 = unnorm9.sum()

print(prob_data9)


# In[37]:


#Summation of Unnormalized Posterior Probability Calculation for Millennials Home Owners & Non-Owners With Student Loans

prob_data10 = unnorm10.sum()

print(prob_data10)


# In[38]:


#Summation of Unnormalized Posterior Probability Calculation for Gen Z Home Owners & Non-Owners With Student Loans

prob_data11 = unnorm11.sum()

print(prob_data11)


# In[39]:


#Summation of Unnormalized Posterior Probability Calculation for Gen Z Home Owners & Non-Owners With No Student Loans

prob_data12 = unnorm12.sum()

print(prob_data12)


# In[40]:


#Normalized Posterior Probability Percentage of Post War Home Owners & Non-Owners With Student Loans

posterior1 = unnorm1 / prob_data1*100

print(posterior1)


# In[41]:


#Normalized Posterior Probability Percentage of Post War Home Owners & Non-Owners With No Student Loans

posterior2 = unnorm2 / prob_data2*100

print(posterior2)


# In[42]:


#Normalized Posterior Probability Percentage of Boomers I Home Owners & Non-Owners With Student Loans

posterior3 = unnorm3 / prob_data3*100

print(posterior3)


# In[43]:


#Normalized Posterior Probability Percentage of Boomers I Home Owners & Non-Owners With No Student Loans

posterior4 = unnorm4 / prob_data4*100

print(posterior4)


# In[44]:


#Normalized Posterior Probability Percentage of Boomers II Home Owners & Non-Owners With Student Loans

posterior5 = unnorm5 / prob_data5*100

print(posterior5)


# In[45]:


#Normalized Posterior Probability Percentage of Boomers II Home Owners & Non-Owners With No Student Loans

posterior6 = unnorm6 / prob_data6*100

print(posterior6)


# In[46]:


#Normalized Posterior Probability Percentage of Gen X Home Owners & Non-Owners With Student Loans

posterior7 = unnorm7 / prob_data7*100

print(posterior7)


# In[47]:


#Normalized Posterior Probability Percentage of Gen X Home Owners & Non-Owners With No Student Loans

posterior8 = unnorm8 / prob_data8*100

print(posterior8)


# In[48]:


#Normalized Posterior Probability Percentage of Millennials Home Owners & Non-Owners With Student Loans

posterior9 = unnorm9 / prob_data9*100

print(posterior9)


# In[49]:


#Normalized Posterior Probability Percentage of Millennials Home Owners & Non-Owners With No Student Loans

posterior10 = unnorm10 / prob_data10*100

print(posterior10)


# In[50]:


#Normalized Posterior Probability Percentage of Gen Z Home Owners & Non-Owners With Student Loans

posterior11 = unnorm11 / prob_data11*100

print(posterior11)


# In[51]:


#Normalized Posterior Probability Percentage of Gen Z Home Owners & Non-Owners With No Student Loans

posterior12 = unnorm12 / prob_data12*100

print(posterior12)


# In[52]:


#Posterior Probability of Home Ownership: Table Summary By Age and Gender

data = [["Post War", "Have Student Loans", posterior1],
        ["Post War", "Do Not Have Student Loans", posterior2],
        ["Boomers I", "Have Student Loans", posterior3],
        ["Boomers I", "Do Not Have Student Loans", posterior4],
        ["Boomers II", "Have Student Loans", posterior5],
        ["Boomers II", "Do Not Have Student Loans", posterior6],
        ["Gen X", "Have Student Loans", posterior7],
        ["Gen X", "Do Not Have Student Loans", posterior8],
        ["Millennials", "Have Student Loans", posterior9],
        ["Millennials", "Do Not Have Student Loans", posterior10],
        ["Gen Z", "Have Student Loans", posterior11],
        ["Gen Z", "Do Not Have Student Loans", posterior12]]

col_names = ["Generation", "Student Loans Status", "Posterior Probability Percentages"]

print(tabulate(data, headers = col_names, tablefmt = "fancy_grid"))

