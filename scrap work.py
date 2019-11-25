# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:34:42 2019

@author: oshapira
"""

%reset

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import collections as coll
import statistics as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
#import datetime
#from dateutil.parser import parse
import seaborn as sns
from scipy import stats
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

import statsmodels.formula.api as smf
import statsmodels.api as sm


os.chdir("C:\\Users\\oshapira\\Desktop\\Analytics\\Uber\\data")
##import dataset and create data frame
df_raw = pd.read_csv('rideshare_kaggle.csv')

df = df_raw.copy()

df['datetime2'] = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S") ##convert datetime to datetime format

#df['time'] = pd.to_datetime(df.datetime2, format="%H:%M:%S").dt.time #extract time from datetime
#df['time_2'] = pd.to_datetime(df.datetime2, format="%H%M%S").dt.time #convert time to integer




df['time_2']= [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df_raw['datetime']] ##create new timestamp column
df['time_2'] = [x.strftime("%H%M%S") for x in df['time_2']] #convert to string format
#df['time_2'] = [str(x) for x in df['time_2']]
df['time_2'] = [int(x) for x in df['time_2']] #convert to integer

df['weekday'] = [x.weekday() for x in df['datetime2']]
df['price_per_mile'] = df['price']/df['distance']


df_sample = df.head(n = 50)


######plot distribution of ride share types

h, axes = plt.subplots (1,2, figsize=(12,4))


Ux=df.name[df.cab_type=='Uber'].unique()
Lx=df.name[df.cab_type=='Lyft'].unique()
Uy = df.name[df.name=='UberXL'].count(),df.name[df.name=='Black'].count(),\
     df.name[df.name=='UberX'].count(),df.name[df.name=='WAV'].count(),\
     df.name[df.name=='Black SUV'].count(),df.name[df.name=='UberPool'].count(),\
     df.name[df.name =='Taxi'].count()

Ly=df.name[df.name=='Shared'].count(),df.name[df.name=='Lux'].count(),\
     df.name[df.name=='Lyft'].count(),df.name[df.name=='Lux Black XL'].count(),\
     df.name[df.name=='Lyft XL'].count(),df.name[df.name=='Lux Black'].count()
     
vis1= sns.barplot(Ux,np.array(Uy),palette='Accent',ax=axes[0])
vis2= sns.barplot(Lx,np.array(Ly),palette='Accent',ax=axes[1])


axes[0].set_title('Number of Uber Rides')
axes[1].set_title('Number of Lyft Rides')
plt.ioff()



types=list(df.name.unique())

types_map = {'Shared':'Shared', 'Lux':'Luxury', 'Lyft':'Standard', 
             'Lux Black XL': 'Luxury', 'Lyft XL':'XL', 'Lux Black':'Luxury', 'UberXL':'XL', 
             'Black':'Luxury', 'UberX':'Standard', 'WAV':'Luxury', 'Black SUV':'Luxury', 'UberPool':'Shared', 
             'Taxi':'Luxury'}

df['ride_type'] = df['name'].map(types_map).fillna(df['name'])

df_sample = df.head(n = 50)

#####filter only on standard type of lyft/uber


df_filter = df.loc[df['ride_type']== 'Standard'].reset_index(drop = True)



columns_to_keep = ['weekday','cab_type','price', 'distance', 'temperature','precipIntensity', 'humidity', 'windSpeed','time', 'time_2', 'price_per_mile']

df_filter = df_filter.filter(items = columns_to_keep)
df_filter = df_filter.dropna() #drop NAs







#import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
fig = go.Figure(data = go.Scatter(
        x = df_filter['distance'],
        y = df_filter['price'],
        mode = 'markers',
        marker_color = df_filter['cab_type']

        ))
fig.show()

plot(fig)

#scatter plot
df_filter.plot(kind = 'scatter', x = 'distance', y = 'price')

#######Must add new traces

from pandas.plotting import scatter_matrix
scatter_matrix(df_filter, alpha=0.2, figsize=(6, 6), diagonal='kde')



#####correlation plot


%matplotlib qt
corr = df_filter.corr()
corr.style.background_gradient(cmap='coolwarm')

f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


##histogram matrix

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.distplot( df_filter["price"] , color="skyblue", ax=axes[0, 0])
sns.distplot( df_filter["distance"] , color="olive", ax=axes[1, 0])
sns.distplot( df_filter["temperature"] , color="gold", ax=axes[0, 1])
sns.distplot( df_filter["time_2"] , color="teal", ax=axes[1, 1])

##separate histograms
sns.distplot(df_filter["price"] , color="skyblue")
sns.distplot(df_filter["distance"] , color="skyblue")
sns.distplot(df_filter["temperature"] , color="skyblue")
sns.distplot(df_filter["time_2"] , color="skyblue")
sns.distplot(df_filter["price_per_mile"] , color="skyblue")



stats.pearsonr(df_filter['price'], df_filter_corr['temperature'])



#########regression modeling#######


def reg_params(variables, weekday, cab_type):
    df_filter2 = df_filter.loc[(df_filter['weekday']== weekday) & (df_filter['cab_type'] == cab_type)]
    X = df_filter2[variables]
    Y = df_filter2['price']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print_model = model.summary()
    return model.summary()
#    return model.pvalues
#    print_model = model.summary()


reg_params(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time_2'], 5, 'Uber')




def reg_params_poly(variables, weekday, cab_type, degree):
    df_filter2 = df_filter.loc[(df_filter['weekday']== weekday) & (df_filter['cab_type'] == cab_type)]
    X = df_filter2[variables]
    Y = df_filter2['price']
    x = np.array(X)
    y = np.array(Y)
    weights = np.polyfit(x, y, degree)
    model = np.poly1d(weights)
    results = smf.ols(formula = 'y ~model(x)', data = df_filter2).fit()
    return results.summary()
#    return model.pvalues


reg_params_poly('windSpeed', 5, 'Uber',2)


def reg_params_plot(variables, weekday, cab_type, degree):
    df_filter2 = df_filter.loc[(df_filter['weekday']== weekday) & (df_filter['cab_type'] == cab_type)]
    fig = go.Figure(data = go.Scatter(        
    x = df_filter2[variables],
    y = df_filter2['price'],
    mode = 'markers'
#    marker_color = df_filter['cab_type']

        ))
    fig.show()

    plot(fig)
    return plot(fig)
    
reg_params_plot('time_2', 5, 'Uber',2)






#df_filter = df_filter.loc[df['weekday']== 2]
X = df_filter[['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time_2']]
Y = df_filter['price']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print_model = model.summary()
print(print_model)



x = df_filter['precipIntensity']



print(print_model)

np.corrcoef(df_filter_corr['price'], df_filter_corr['windSpeed'])

















##with sklearn
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)






##sunday = 6, monday = 0
df_weekday = df.loc[df['weekday']== 0]






































































from datetime import datetime
# Creating a datetime object so we can test.
a = datetime.now()

# Converting a to string in the desired format (YYYYMMDD) using strftime
# and then to int.
a = int(a.strftime('%Y%m%d'))



df_raw.datetime.dtype

pd.Datetimeindex(df_raw['datetime']).hour

time = pd.Timestamp.time(datetimeObj)



df_raw['time'].dtype

df_raw['datetime2'].dtype


sample = df_raw['datetime'][1]

dt = parse(sample)

print(dt.time())

datetimeObj = datetime.strptime(df_raw['datetime'][1], '%Y-%m-%d %H:%M:%S')

datetimeObj.time()

t = datetime.time(1,3,4)

print(t)

int(obj)

obj = datetimeObj.strftime("%H%M%S")

print(obj)
int(obj)

print(type(datetimeObj))

print(type(sample))