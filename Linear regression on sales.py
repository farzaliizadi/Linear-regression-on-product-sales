"""Created on Tue Mar 19 15:41:06 2019@author: Izadi   """
# 1. loading modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
os.chdir('D:\\mining code')  
os.getcwd()

#2. Readind csv file. Since file one Unknown column , I drop it by index_col=0
df = pd.read_csv('SALES.csv', index_col=0)
df.shape
#3 (2000, 4) 
df.head()
#4. This gives us all necessary information about the data:Col-Names,N0. of ovsevations and type of columns 
df.info()
#5 This gives a good statistical summary od tge data. Max, min, median, mean, std and 25%, 50% and 75% quantiles. 
df.describe()
#6. To check if there is any null which is not
df1 = df.isnull().sum()
df1   # None of columns have missing values.
#6. We drop Target variable and took the rest as X
X = df.drop(['sales'], axis=1)
#7. We pick up target variavles
y = df['sales']
#8. Here we start excuting the recursive feature eliminion
features = X.columns.values

results = []
lr =  LinearRegression()
estimator = LinearRegression()
selector = RFE(estimator,3,step=1)
selcetor = selector.fit(X,y)
selector.support_                   #9.I got 3 True, so all features are necessary
selector.ranking_                   #10 ranking = array[1, 1, 1]

for i in range(1,len(X.iloc[0])+1):
    selector = RFE(lr, n_features_to_select=i, step=1)
    selector.fit(X,y)
    r2 = selector.score(X,y)
    selected_features = features[selector.support_]
    msr = mean_squared_error(y, selector.predict(X))
    results.append([i, r2, msr, ','.join(selected_features)])
#9. This loops also works correctly.  
r2 
selected_features 
msr
#10. Cross validation
from sklearn.model_selection import cross_val_score
crossValscore = cross_val_score(lr,X, y, cv=20)
crossValscore
from sklearn.model_selection import ShuffleSplit
n_samples =df.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(lr, df, df.sales, cv=cv)
cross_val_score(estimator, X, y,cv =5)
    
#11. Here I wish to find the heat map of variables'
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10)
    cax = ax1.imshow(df.corr('kendall'), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('crd Feature Correlation')
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=20)
    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
    
    correlation_matrix(df)
#12. Here I got all the diffirent corellations.By default corr() is pearson 
df.corr(method='pearson')       
df.corr(method='spearman')
df.corr(method='kendall')

from pylab import rcParams
rcParams['figure.figsize'] = 30, 20
#13. Here we got pair plot between Tager variable and the other 3 independent features.
sns.pairplot(df, x_vars=['TV','radio', 'newspaper'],y_vars='sales'  , size=7, aspect=0.7) 
plt.show()
sns.pairplot(df, x_vars=['TV','radio', 'newspaper'], y_vars='sales' , size=7, aspect=0.7 , kind='reg') 
plt.show()
plt.scatter(df.TV,df.sales)
plt.show()
plt.scatter(df.radio, df.sales)
plt.show()
plt.scatter(df.newspaper, df.sales)
plt.show()



