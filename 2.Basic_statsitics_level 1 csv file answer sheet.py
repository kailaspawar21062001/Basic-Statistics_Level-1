# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:08:15 2023

@author: kailas
"""
"D:\data science assignment\Assignments\#Basic Statistics_Level-1\Q7.csv"




import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.stats as stats

             

#######################################################################################################

                       @@@question 6 @@@


def expected_value(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)
    return (values * weights).sum() / weights.sum()
c_count = [1,4,3,5,6,2]
ch_prob = [0.015,0.20,0.65,0.005,0.01,0.120]
expected_value(c_count, ch_prob)
3.09

############################################################################
                      @@@question 7 @@@


q7 = pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/Q7.csv")
q7

q7.describe()

stats.median(q7["Points"])
3.6950000000000003
q7.median()

q7.mode()

stats.mode(q7['Points'])
3.07
stats.mode(q7['Score'])
3.44
stats.mode(q7['Weigh'])
17.02
q7.var()

q7.rename(columns={'Unnamed: 0':'Cars'}, inplace = True)
q7

q7.set_index(('Cars'), inplace = True)
q7

plt.hist(q7["Points"], bins = 10, edgecolor= 'black')
plt.show()

plt.boxplot(x = 'Points', data =q7)
plt.xlabel('Points')
plt.ylabel('Density')
plt.savefig("PointsInferences.png")
plt.show()

plt.hist(q7["Score"], bins = 20, edgecolor = 'y')
plt.show()

plt.boxplot(x = 'Score', data= q7)
plt.xlabel('Scores')
plt.ylabel('Density')
plt.savefig("ScoresInferences.png")
plt.show()

plt.hist(q7["Weigh"], bins=20, edgecolor = 'red')
plt.show()

plt.boxplot(x= "Weigh", data = q7)
plt.xlabel('Weigh')
plt.ylabel('Density')
plt.savefig("WeighInferences.png")
plt.show()

plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Points"])
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Score"])
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Weigh"])
plt.yticks(fontsize=14)
plt.show()

 
 
      

###############################################################################
                         @@@question 8 @@@


weigh = [108,110,123,134,135,145,167,187,199]
probs = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]
expected_value(weigh, probs)
145.33333333333331
ch = 1/9
ch
0.1111111111111111



                 

 ####################################################################################   
                            @@@Question 9 @@@


q9a = pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/Q9_a (1).csv", index_col = 'Index')
q9a

print('For Cars Speed', "Skewness value=", np.round(q9a.speed.skew(),2), 'and' , 'Kurtosis value=', np.round(q9a.speed.kurt(),2))

print('Skewness value =', np.round(q9a.dist.skew(),2),'and', 'Kurtosis value =', np.round(q9a.dist.kurt(),2), 'for Cars Distance')

q9b =pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/Q9_b.csv")
q9b


q9b.rename(columns = {'Unnamed: 0':'Index'}, inplace = True)
q9b


q9b


print('For SP Skewness =', np.round(q9b.SP.skew(),2), 'kurtosis =', np.round(q9b.SP.kurt(),2))

print('For WT Skewness =', np.round(q9b.WT.skew(),2), 'Kurtosis =', np.round(q9b.WT.kurt(),2))


######################################################################################
                             @@@question 11 @@@


from scipy import stats
conf_94 =stats.t.interval(alpha = 0.97, df=1999, loc=200, scale=30/np.sqrt(2000))
print(np.round(conf_94,0))
print(conf_94)

conf_94 =stats.t.interval(alpha = 0.94, df=1999, loc=200, scale=30/np.sqrt(2000))
print(np.round(conf_94,0))
print(conf_94)
[199. 201.]
(198.7376089443071, 201.2623910556929)

conf_98= stats.t.interval(alpha = 0.98, df = 1999, loc = 200, scale=30/np.sqrt(2000))
print(np.round(conf_98,0))
print(conf_98)
[198. 202.]
(198.4381860483216, 201.5618139516784)

conf_96 = stats.t.interval( alpha = 0.96, df = 1999 , loc = 200 , scale = 30/np.sqrt(2000))
print(np.round(conf_96,0))
print(conf_96)
[199. 201.]
(198.6214037429732, 201.3785962570268)

conf_z_94 = stats.norm.interval(0.94, loc = 200, scale = 30/np.sqrt(2000))
np.round(conf_z_94,0)
array([199., 201.])

conf_z_96 = stats.norm.interval(0.96, loc = 200, scale = 30/np.sqrt(2000))
np.round(conf_z_94,0)
array([199., 201.])

conf_z_98 =  stats.norm.interval(0.98, loc=200,scale=30/np.sqrt(2000))
np.round(conf_z_98,0)
array([198., 202.])
 
stats.t.ppf(0.03,df=1999)
-1.8818614764780115
stats.t.ppf(0.01,df=1999)
-2.3282147761069725
stats.t.ppf(0.02,df=1999)
-2.055089962825778



###################################################################################
                   @@@question 12 @@@


q12 = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
stat.mean(q12)
41
stat.median(q12)
40.5
stat.variance(q12)
25.529411764705884
stat.stdev(q12)
5.05266382858645




q12_df = pd.DataFrame({'students':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                    'marks':(q12)})
q12_df

q12_df.describe()

q12_df.set_index('students')




######################################################################################
                        @@@question 20 @@@


q20 = pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/Cars (2).csv")
q20


from scipy import stats
q20.describe()

Prob_MPG_greater_than_38 = np.round(1 - stats.norm.cdf(38, loc= q20.MPG.mean(), scale= q20.MPG.std()),3)
print('P(MPG>38)=',Prob_MPG_greater_than_38)

prob_MPG_less_than_40 = np.round(stats.norm.cdf(40, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('P(MPG<40)=',prob_MPG_less_than_40)

prob_MPG_greater_than_20 = np.round(1-stats.norm.cdf(20, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('p(MPG>20)=',(prob_MPG_greater_than_20))

prob_MPG_less_than_50 = np.round(stats.norm.cdf(50, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('P(MPG<50)=',(prob_MPG_less_than_50))

prob_MPG_greaterthan20_and_lessthan50= (prob_MPG_less_than_50) - (prob_MPG_greater_than_20)
print('P(20<MPG<50)=',(prob_MPG_greaterthan20_and_lessthan50))




######################################################################################
                            @@@Question 21@@@


q21a = pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/Cars (2).csv")
q21a


import numpy as np
import matplotlib.pyplot as plt

mean, cov = [0, 0], [(1, .6), (.6, 1)]
x, y = np.random.multivariate_normal(mean, cov, 100).T
y += x + 1

f, ax = plt.subplots(figsize=(6, 6))

ax.scatter(x, y, c=".3")
ax.set(xlim=(-3, 3), ylim=(-3, 3))

# Plot your initial diagonal line based on the starting
# xlims and ylims.
diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

def on_change(axes):
    # When this function is called it checks the current
    # values of xlim and ylim and modifies diag_line
    # accordingly.
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    diag_line.set_data(x_lims, y_lims)

# Connect two callbacks to your axis instance.
# These will call the function "on_change" whenever
# xlim or ylim is changed.
ax.callbacks.connect('xlim_changed', on_change)
ax.callbacks.connect('ylim_changed', on_change)

plt.show()

plt.hist(q21a["MPG"], bins = 20, edgecolor=  'black')
plt.show()

plt.boxplot(x= 'MPG', data =q21a)
plt.show()

import statsmodels.api as sm
sm.qqplot(q21a['MPG'])
plt.xlabel('MPG', color ='red')
plt.savefig('MPG of cars.png')
plt.show()


import scipy.stats as stats
stats.probplot(q21a['MPG'], dist="norm", plot=plt)
plt.xlabel('MPG', color ='red')
plt.savefig('MPG of cars.png')
plt.show()

sn.distplot(q21a['MPG'],kde=True, bins =10)
plt.show()


q21b = pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/wc-at.csv")
q21b


plt.hist(q21b['Waist'], edgecolor= 'red')
plt.show()

plt.boxplot(x = 'Waist', data= q21b)
plt.title("Waist")
plt.savefig('Waist.png')
plt.show()

sn.distplot(q21b['Waist'], 
             bins=10,
            kde = True
            )
plt.show()


import statsmodels.api as sm
sm.qqplot(q21b['Waist'])
plt.show()


stats.probplot(q21b['Waist'], dist = 'norm', plot = plt)
plt.xlabel('Waist', color= 'red')
plt.savefig('Waist.png')
plt.show()

sn.distplot(q21b['AT'], bins =10, kde=True)
plt.show()


import statsmodels.api as sm
sm.qqplot(q21b['AT'])
plt.show()


stats.probplot(q21b['AT'], dist = 'norm', plot = plt)
plt.xlabel('AT', color= 'red')
plt.savefig('AT.png')
plt.show()





########################################################################################
                      @@@Question 22 @@@

# z value for 90% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.05),4))

# z value for 94% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.03),4))

# z value for 60% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.2),4))

# t score for 95% confidence interval
print('T score for 95% Confidence Interval =',np.round(stats.t.ppf(0.025,df=24),4))

# t value for 94% confidence interval
print('T score for 94% Confidence Inteval =',np.round(stats.t.ppf(0.03,df=24),4))

# t value for 99% Confidence Interval
print('T score for 95% Confidence Interval =',np.round(stats.t.ppf(0.005,df=24),4))




########################################################################################
                               @@@question 23 @@@

from scipy import stats
from scipy.stats import norm
# t scores of 95% confidence interval for sample size of 25
stats.t.ppf(0.975,24)  # df = n-1 = 24
2.0638985616280205
# t scores of 96% confidence interval for sample size of 25
stats.t.ppf(0.98,24)
2.1715446760080677
# t scores of 99% confidence interval for sample size of 25
stats.t.ppf(0.995,24)
2.796939504772804
 

 

#######################################################################################           
                              @@@Question 24 @@@


from scipy import stats
from scipy.stats import norm

x_bar = 260
pop_mean = 270
t_value = (260-270)/(90/np.sqrt(18))
t_value
-0.4714045207910317
1-stats.t.cdf(abs(t_value),df = 17)
0.32167253567098353
