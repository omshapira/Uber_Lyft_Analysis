# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:27:14 2019

@author: oshapira
"""
%reset

import numpy as np
import pandas as pd
import os




#import collections as coll
#import statistics as st
#from sklearn.metrics import mean_squared_error, r2_score
#from mpl_toolkits.mplot3d import Axes3D

#import datetime
#from dateutil.parser import parse


os.chdir("C:\\Users\\oshapira\\Desktop\\Analytics\\Uber\\data")
##import dataset and create data frame
df_raw = pd.read_csv('rideshare_kaggle.csv')

df = df_raw.copy()



###initial filtering of columns desired for analysis
columns_to_keep = ['timestamp', 'hour', 'datetime', 'source', 'destination', 'cab_type', 'name', 'price', 'distance', 
                   'surge', 'temperature','precipIntensity', 'humidity', 'windSpeed']

df = df.filter(items = columns_to_keep)
df = df.dropna() #drop NAs


##create new datetime column that is in date_time format
df['datetime_2'] = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S") ##convert datetime to datetime format

##add variable for day of week for given ride
##sunday = 6, monday = 0
df['weekday'] = [x.weekday() for x in df['datetime_2']]



from datetime import datetime

####extract precise hour and minute of day from datetime variable
df['time']= [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['datetime']] ##create new timestamp column
df['time'] = [x.strftime("%H%M%S") for x in df['time']] #convert to string format
#df['time_2'] = [str(x) for x in df['time_2']]
df['time'] = [int(x) for x in df['time']] #convert to integer


######plot distribution of ride share types

import matplotlib.pyplot as plt
%matplotlib qt
import seaborn as sns
h, axes = plt.subplots (1,2, figsize=(12,4))


Ux=df.name[df.cab_type=='Uber'].unique()
Lx=df.name[df.cab_type=='Lyft'].unique()
Uy = df.name[df.name=='UberXL'].count(),df.name[df.name=='Black'].count(),\
     df.name[df.name=='UberX'].count(),df.name[df.name=='WAV'].count(),\
     df.name[df.name=='Black SUV'].count(),df.name[df.name=='UberPool'].count()

Ly=df.name[df.name=='Shared'].count(),df.name[df.name=='Lux'].count(),\
     df.name[df.name=='Lyft'].count(),df.name[df.name=='Lux Black XL'].count(),\
     df.name[df.name=='Lyft XL'].count(),df.name[df.name=='Lux Black'].count()
     
vis1= sns.barplot(Ux,np.array(Uy),palette='Accent',ax=axes[0])
vis2= sns.barplot(Lx,np.array(Ly),palette='Accent',ax=axes[1])


axes[0].set_title('Number of Uber Rides')
axes[1].set_title('Number of Lyft Rides')
plt.ioff()


###create column for ride types so that Uber and Lyft can be more comparable
types=list(df.name.unique())

types_map = {'Shared':'Shared', 'Lux':'Luxury', 'Lyft':'Standard', 
             'Lux Black XL': 'Luxury', 'Lyft XL':'XL', 'Lux Black':'Luxury', 'UberXL':'XL', 
             'Black':'Luxury', 'UberX':'Standard', 'WAV':'Luxury', 'Black SUV':'Luxury', 'UberPool':'Shared', 
             'Taxi':'Luxury'}

df['ride_type'] = df['name'].map(types_map).fillna(df['name'])


df_sample = df.head(n = 100)




##########filter data for regression analysis#########


def df_regression_filter(variables, cab_type, ride_type):
    df_filter = df.loc[(df['ride_type'].isin(ride_type)) & (df['cab_type'].isin(cab_type))]
    df_filter = df_filter.filter(items = variables)   
    return df_filter


#######create scatterplot matrix
df_plot = df_regression_filter(['price','distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time'], ['Uber'], ['Shared'])


from pandas.plotting import scatter_matrix
    
scatter_matrix(df_plot, alpha=0.2, figsize=(len(df_plot.columns), len(df_plot.columns)), diagonal='kde')
plt.show()



#########regression modeling#######


########onehot encoding for "source"

#df_reg = df_regression(['price','hour','source','distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time'], 'Uber', 'Shared')

#https://towardsdatascience.com/multiple-linear-regression-in-four-lines-of-code-b8ba26192e84
#https://stackoverflow.com/questions/45740920/statsmodels-how-can-i-get-statsmodel-to-return-the-pvalue-of-an-ols-object
#
#X = df_reg.iloc[:,1:].values
#y = df_reg.iloc[:,0].values
#
#from sklearn.preprocessing import LabelEncoder
#
#labelencoder_X = LabelEncoder()
#X[:,1] = labelencoder_X.fit_transform(X[:,1])
#
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [0])
#
#
#X = onehotencoder.fit_transform(X).toarray()
#
#


from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm

#######function that does backwards elimination to remove insignificant features

def linear_params_test(variables, cab_type, ride_type, sigvalue):
    variables = variables
    variables.insert(0,'price')
    df_filter = df_regression_filter(variables, cab_type, ride_type)
#    X = df_filter.filter(items = variables) 
    
    X = df_filter.loc[:, df_filter.columns != 'price']
#    X = df_filter[variables]    
    Y = df_filter['price']
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#    model = sm.OLS(y_train, X_train).fit()
#    print_model = model.summary()    
    lrmodel = sm.OLS(y_train, X_train).fit()
    pVals = lrmodel.pvalues
    pVals = pVals.to_frame().reset_index(drop = False)
    pVals.columns= ['index', 'pvalue']
    while max(pVals['pvalue'])>sigvalue:
        for p in range(0, len(pVals['pvalue'])): 
            p_value = pVals['pvalue'][p]
            i_max = pVals['pvalue'].values.argmax()
            column_drop = pVals['index'][i_max]
        X_train.drop([column_drop], axis = 1, inplace = True)
        X_test.drop([column_drop], axis = 1, inplace = True)
        lrmodel = sm.OLS(y_train,X_train).fit()
        pVals= lrmodel.pvalues
        pVals = pVals.to_frame().reset_index(drop = False)
        pVals.columns= ['index', 'pvalue']
    
    lrmodel = sm.OLS(y_train, X_train).fit()

#    summary = lrmodel.summary()
#    return summary
    return lrmodel



def linear_params_summary(variables, cab_type, ride_type, sigvalue):
    lrmodel = linear_params_test(variables, cab_type, ride_type, sigvalue)
    summary = lrmodel.summary()
    return summary

linear_params_summary(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Shared', 'Luxury', 'Standard'], 0.05)
linear_params_summary(['distance'], ['Uber', 'Lyft'], ['Shared', 'Luxury', 'Standard'], 0.05)


##linear regression line for standard rides is best, particularly for Lyft
##by ride type
linear_params_summary(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Standard'], 0.05)
linear_params_summary(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Luxury'], 0.05)
linear_params_summary(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Shared'], 0.05)

#by cab Lyft vs. Uber for Standard ride types
linear_params_summary(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber'], ['Standard'], 0.05)
linear_params_summary(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Lyft'], ['Standard'], 0.05)




from sklearn.linear_model import LinearRegression ###plot test data

def lin_reg_cab_df(cab_type, ride_type):
    
    features = ['price', 'distance', 'ride_type', 'cab_type']
    df_filter = df.filter(items = features) 
    df_filter = df_filter.loc[(df_filter['ride_type'].isin(ride_type))]
    df_filter_cab = df_filter.loc[(df_filter['cab_type'] == cab_type)]

    X_cab = df_filter_cab.loc[:,df_filter_cab.columns == 'distance']
    Y_cab = df_filter_cab.loc[:,df_filter_cab.columns == 'price']

    X_train_cab, X_test_cab, y_train_cab, y_test_cab = train_test_split(X_cab, Y_cab, test_size = 0.2, random_state = 0)
    
    model_cab = LinearRegression()
    model_cab.fit(X_train_cab, y_train_cab)
    
    X_test_cab_df = pd.DataFrame(X_test_cab['distance'].reset_index(drop = True))
    y_test_cab_df = pd.DataFrame(y_test_cab['price'].reset_index(drop = True))
    cab_predicted = pd.DataFrame(model_cab.predict(X_test_cab))
    
    cab_df = pd.concat([X_test_cab_df, y_test_cab_df, cab_predicted], axis = 1)
    cab_df.columns = ['distance', 'actual_price', 'predicted_price']
    
    return cab_df




import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go

def plot_lin_reg(ride_type):
    uber_df = lin_reg_cab_df('Uber', ride_type)
    lyft_df = lin_reg_cab_df('Lyft', ride_type)

    trace0 = go.Scatter(
            x = uber_df['distance'], 
            y  = uber_df['actual_price'], 
            mode = 'markers',
            name = 'Uber Actual Prices'
            )
    trace1 = go.Scatter(
            x = lyft_df['distance'], 
            y  = lyft_df['actual_price'], 
            mode = 'markers',
            name = 'Lyft Actual Prices',
            )
    
    trace2 = go.Scatter(
            x = uber_df['distance'], 
            y  = uber_df['predicted_price'], 
            mode = 'lines',
            name = 'Uber Regression Line'
            )
    
    trace3 = go.Scatter(
            x = lyft_df['distance'], 
            y  = lyft_df['predicted_price'], 
            mode = 'lines',
            name = 'Lyft Regression Line'
            )

    data = [trace0, trace1, trace2, trace3]
    return plot(data)


plot_lin_reg(['Standard'])



from scipy.stats import f as fisher_f
from sklearn import metrics









###############significance in regression difference Chow test


def lin_reg_analysis(ride_type):
    ###Both Lyft and Uber
    
    features = ['price', 'distance', 'ride_type', 'cab_type']
    df_filter = df.filter(items = features) 
    df_filter_all = df_filter.loc[(df_filter['cab_type'].isin(['Lyft','Uber']) & (df_filter['ride_type'].isin(ride_type)))]
    X_all = df_filter_all.loc[:,df_filter_all.columns == 'distance']
    Y_all = df_filter_all.loc[:,df_filter_all.columns == 'price']
    #X_all = X_all['distance'].values
    #Y_all = Y_all['price'].values
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, Y_all, test_size = 0.2, random_state = 0)
    
    ##all regression
    model_all = LinearRegression(fit_intercept = True)
    #model_all.fit(X_test_all[:, np.newaxis], y_test_all)
    model_all.fit(X_train_all, y_train_all)
    
    slope_all = model_all.coef_[0]
    intercept_all = model_all.intercept_
    predict_all = model_all.predict(X_test_all)
    r2_test_all = model_all.score(X_test_all, y_test_all)
    MAE_all = metrics.mean_absolute_error(predict_all, y_test_all)
    MSE_all = metrics.mean_squared_error(predict_all, y_test_all)
    RMSE_all = np.sqrt(metrics.mean_squared_error(predict_all, y_test_all))
    
    ##lyft
    features = ['price', 'distance', 'ride_type', 'cab_type']
    df_filter = df.filter(items = features) 
    df_filter_lyft = df_filter.loc[(df_filter['cab_type'].isin(['Lyft']) & (df_filter['ride_type'].isin(ride_type)))]
    X_lyft = df_filter_lyft.loc[:,df_filter_lyft.columns == 'distance']
    Y_lyft = df_filter_lyft.loc[:,df_filter_lyft.columns == 'price']
    #X_lyft = X_lyft['distance'].values
    #Y_lyft = Y_lyft['price'].values
    X_train_lyft, X_test_lyft, y_train_lyft, y_test_lyft = train_test_split(X_lyft, Y_lyft, test_size = 0.2, random_state = 0)
    
    ##lyft regression
    model_lyft = LinearRegression(fit_intercept = True)
    #model_lyft.fit(X_test_lyft[:, np.newaxis], y_test_lyft)
    model_lyft.fit(X_test_lyft, y_test_lyft)
    
    slope_lyft = model_lyft.coef_[0]
    intercept_lyft = model_lyft.intercept_
    predict_lyft = model_lyft.predict(X_test_lyft)
    r2_test_lyft = model_lyft.score(X_test_lyft, y_test_lyft)
    MAE_lyft = metrics.mean_absolute_error(predict_lyft, y_test_lyft)
    MSE_lyft = metrics.mean_squared_error(predict_lyft, y_test_lyft)
    RMSE_lyft = np.sqrt(metrics.mean_squared_error(predict_lyft, y_test_lyft))
    
    
    ###########uber
    features = ['price', 'distance', 'ride_type', 'cab_type']
    df_filter = df.filter(items = features) 
    df_filter_uber = df_filter.loc[(df_filter['cab_type'].isin(['Uber']) & (df_filter['ride_type'].isin(ride_type)))]
    X_uber = df_filter_uber.loc[:,df_filter_uber.columns == 'distance']
    Y_uber = df_filter_uber.loc[:,df_filter_uber.columns == 'price']
    #X_uber = X_uber['distance'].values
    #Y_uber = Y_uber['price'].values
    X_train_uber, X_test_uber, y_train_uber, y_test_uber = train_test_split(X_uber, Y_uber, test_size = 0.2, random_state = 0)
    
    ##uber regression
    model_uber = LinearRegression(fit_intercept = True)
    #model_uber.fit(X_test_uber[:, np.newaxis], y_test_uber)
    model_uber.fit(X_test_uber, y_test_uber)
    
    slope_uber = model_uber.coef_[0]
    intercept_uber = model_uber.intercept_
    predict_uber = model_uber.predict(X_test_uber)
    r2_test_uber = model_uber.score(X_test_uber, y_test_uber)
    MAE_uber = metrics.mean_absolute_error(predict_uber, y_test_uber)
    MSE_ubrer = metrics.mean_squared_error(predict_uber, y_test_uber)
    RMSE_uber = np.sqrt(metrics.mean_squared_error(predict_uber, y_test_uber))

    # perform the Chow test:
    
    SSE_all = sum((np.asarray(y_test_all - predict_all))*(np.asarray(y_test_all - predict_all)))
    SSE_lyft = sum((np.asarray(y_test_lyft - predict_lyft))*(np.asarray(y_test_lyft - predict_lyft)))
    SSE_uber = sum((np.asarray(y_test_uber - predict_uber))*(np.asarray(y_test_uber - predict_uber)))
    
    
    N = len(y_test_all['price'])
    deg_freedom_all = len(y_test_all['price']) - 1 
    deg_freedom_1_lyft = len(y_test_lyft['price']) - 1
    deg_freedom_2_uber = len(y_test_uber['price']) - 1
    
    k = 2 # one dimensional regression - slope and intercept
    numerator = (SSE_all - (SSE_lyft + SSE_uber))/k
    denominator = (SSE_lyft + SSE_uber)/(N - 2*k) #both regression have 2 degrees of freedom
    
    f_statistics = numerator/denominator #calculate f-statistic significant value for each months' regressions
    
    alpha = 0.05
    p_value = fisher_f.cdf(f_statistics, 2, N-2*2) ##calculate critical value for degrees 2 and 2
    
    if p_value > alpha:
        chow_test = str('The Chow test shows that there is a statistically significant different between the Uber and Lyft regression lines')
    else:
        chow_test = str('The Chow test shows that there is not a statistically significant different between the Uber and Lyft regression lines')
    
    
    reg_analysis = print(f"The linear regression line for Uber is y = " + str(round(slope_uber[0],2))+ "x + " + str(round(intercept_uber[0],2)) + " with a r-squared coefficient value of "  
                        + str(round(r2_test_uber,2)) +  " a Mean Absolute error of " + str(round(MAE_uber,2)) +  " and a Root Mean Squared Error of " + str(round(RMSE_uber,2)) + "\n" +
    "The linear regression line for Lyft is y = " + str(round(slope_lyft[0],2))+"x + " + str(round(intercept_lyft[0],2))+ " with a r-squared coefficient value of " 
    + str(round(r2_test_lyft,2)) +  " a Mean Absolute error of " + str(round(MAE_lyft,2)) +  " and a Root Mean Squared Error of " + str(round(RMSE_lyft,2)) + "\n" +
    "The linear regression line for both Uber and Lyft combined is is y = " + str(round(slope_all[0],2))+"x + " + str(round(intercept_all[0],2)) + " with a r-squared coefficient value of " + str(round(r2_test_all,2))  + "\n" +
    chow_test)
    
    return reg_analysis


lin_reg_analysis(['Standard'])




############different degrees with regression

import statsmodels.formula.api as smf


def reg_params_poly(variables, cab_type, ride_type, degree):
    df_filter = df.loc[(df['ride_type'].isin(ride_type)) & (df['cab_type'].isin(cab_type))]
    X = df_filter[variables]
    Y = df_filter['price']
    x = np.array(X)
    y = np.array(Y)
    weights = np.polyfit(x, y, degree)
    model = np.poly1d(weights)
    results = smf.ols(formula = 'y ~model(x)', data = df_filter).fit()
#    return results.summary()
    return results


reg_params_poly('distance', ['Uber', 'Lyft'], ['Standard'],3)

reg_params_poly('distance', ['Uber', 'Lyft'], ['Standard'],2)
reg_params_poly('distance', ['Uber', 'Lyft'], ['Standard'],4)
reg_params_poly('distance', ['Uber', 'Lyft'], ['Standard'],5)
reg_params_poly('distance', ['Uber', 'Lyft'], ['Standard'],6)


def reg_poly_predict(variables, cab_type, ride_type, degree):
    df_filter = df.loc[(df['ride_type'].isin(ride_type)) & (df['cab_type'].isin(cab_type))]
    X = df_filter[variables]
    Y = df_filter['price']
    x = np.array(X)
    y = np.array(Y)
    weights = np.polyfit(x, y, degree)
    model = np.poly1d(weights)
    results = smf.ols(formula = 'y ~model(x)', data = df_filter).fit()
    return results.summary()
#    return model.pvalues



#
#def reg_params_plot(variables, weekday, cab_type, degree):
#    df_filter2 = df_filter.loc[(df_filter['weekday']== weekday) & (df_filter['cab_type'] == cab_type)]
#    fig = go.Figure(data = go.Scatter(        
#    x = df_filter2[variables],
#    y = df_filter2['price'],
#    mode = 'markers'
##    marker_color = df_filter['cab_type']
#
#        ))
#    fig.show()
#
#    plot(fig)
#    return plot(fig)
#    
#reg_params_plot('time_2', 5, 'Uber',2)





######################KNN REGRESSION##################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

def df_regression(variables, cab_type, ride_type):
    df_filter = df.loc[(df['ride_type']== ride_type) & (df['cab_type'] == cab_type)]
    df_filter = df_filter.filter(items = variables)   
    return df_filter


#######create scatterplot matrix
#df_knn = df_regression(['price','distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time'], 'Uber', 'Shared')
   

from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt

def knn_regression_filter(variables, cab_type, ride_type):
    rmse_val = [] #to store rmse values for different k
    k_list = []
    df_filter = df.loc[(df['ride_type'].isin(ride_type)) & (df['cab_type'].isin(cab_type))]
    #    X = df_filter.filter(items = variables) 
    X = df_filter[variables]    
    Y = df_filter['price']
#    X = sm.add_constant(X)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    for K in range(3,21,2):
        K = K
        model = neighbors.KNeighborsRegressor(n_neighbors = K)
        model.fit(X_train, y_train)  #fit the model
        pred=model.predict(X_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        r2 = model.score(X_test, y_test)
        rmse_val.append(error) #store rmse values
        k_list.append(K)
        printed = print('RMSE value for k= ' , K , 'is:', error, 'and R-square is ', str(round(r2,2)))
    curve = pd.DataFrame(rmse_val,k_list).reset_index(drop = False) #elbow curve 
    curve.columns = ['k_value', 'rmse']
    k_array = np.asarray(k_list)
    rmse_val_array = np.asarray(rmse_val)
    
    fig = go.Figure(data = go.Scatter(x= curve['k_value'], y=curve['rmse']))
    fig.update_layout(
            title = 'Accuracy per number of K Neighbors',
            xaxis = dict(
                    tick0 = 3,
                    dtick = 2,
                    title_text = '# of Neighbors'
                    ),
            yaxis = dict(
                    title_text = 'Root Mean Squared Error')
                )
        
    return plot(fig)



knn_regression_filter(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Standard'])


##higher performance when only using 'distance' feature
knn_regression_filter(['distance'], ['Uber', 'Lyft'], ['Standard'])



#################Random Forests

def df_regression_filter(variables, cab_type, ride_type):
    df_filter = df.loc[(df['ride_type'].isin(ride_type)) & (df['cab_type'].isin(cab_type))]
    df_filter = df_filter.filter(items = variables)   
    return df_filter



from sklearn.ensemble import RandomForestRegressor

def partition_dataset(variables, cab_type, ride_type):
    variables = variables
    variables.insert(0,'price')
    df_filter = df_regression_filter(variables, cab_type, ride_type)    
    X = df_filter.loc[:, df_filter.columns != 'price']
    Y = df_filter['price']
#    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)   
    return X_train, X_test, y_train, y_test




def create_rf_model(variables, cab_type, ride_type):
    X_train, X_test, y_train, y_test = partition_dataset(variables, cab_type, ride_type)
    rf_model = RandomForestRegressor(n_estimators = 10, random_state = 0)
#    return X_train, X_test, y_train, y_test
    return rf_model
test = partition_dataset(['distance'], ['Uber', 'Lyft'], ['Standard'])
    
    
def rf_model_predictions(variables, cab_type, ride_type):
    X_train, X_test, y_train, y_test = partition_dataset(variables, cab_type, ride_type)
    rf_model = create_rf_model(variables, cab_type, ride_type)
    rf_model.fit(X_train, y_train)  #fit the model    
    rf_pred=rf_model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,rf_pred)) #calculate rmse
    r2 = rf_model.score(X_test, y_test)
#    rmse_val.append(error) #store rmse values
    score = print('R-square is ', str(round(r2,2)))

    return r2

rf_model_predictions(['distance'], ['Uber', 'Lyft'], ['Standard'])




rf_model_predictions(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Standard'])

rf_model_predictions(['distance'], ['Uber', 'Lyft'], ['Standard'])


def rf_model_feature_importance(variables, cab_type, ride_type):
    feature_list = variables
    X_train, X_test, y_train, y_test = partition_dataset(variables, cab_type, ride_type)
    rf_model = create_rf_model(variables, cab_type, ride_type)
    rf_model.fit(X_train, y_train)  #fit the model 
    importances = list(rf_model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    feature_ranking = [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    return feature_ranking


rf_model_feature_importance(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Standard'])








from sklearn.ensemble import RandomForestRegressor


variables = ['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday']
variables.insert(0,'price')
df_filter = df_regression_filter(variables, ['Uber', 'Lyft'], ['Standard'])    
X = df_filter.loc[:, df_filter.columns != 'price']
Y = df_filter['price']
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)   
#return X_train, X_test, y_train, y_test

rf_model = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_model.fit(X_train, y_train)  #fit the model   
rf_pred=rf_model.predict(X_test) #make prediction on test set
#rf_model_predictions(['distance'], ['Uber', 'Lyft'], ['Standard'])



error = sqrt(mean_squared_error(y_test,rf_pred)) #calculate rmse
r2 = rf_model.score(X_test, y_test)
#    rmse_val.append(error) #store rmse values
score = print('R-square is ', str(round(r2,2)))

#rf_model(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'], ['Standard'])
#
#rf_model_predictions(['distance'], ['Uber', 'Lyft'], ['Standard'])


importances = list(rf_model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(variables, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


































plt.plot(range(3,13,2), grouped_pred_2017['True'], color='red', linestyle='dashed',
marker='o', markerfacecolor='black', markersize=10)
plt.title('Error Rate vs. k for Starbucks')
plt.xlabel('number of neighbors: k')
plt.ylabel('Error Rate')


import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


rmse_val = [] #to store rmse values for different k
k_list = []
df_filter = df.loc[(df['ride_type'].isin(ride_type_2)) & (df['cab_type'].isin(cab_type_2))]
#    X = df_filter.filter(items = variables) 
X = df_filter[variables_2]    
Y = df_filter['price']
#    X = sm.add_constant(X)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
for K in range(3,21,2):
    K = K
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    r2 = model.score(X_test, y_test)
    rmse_val.append(error) #store rmse values
    k_list.append(K)
    printed = print('RMSE value for k= ' , K , 'is:', error)

curve = pd.DataFrame(rmse_val,k_list).reset_index(drop = False) #elbow curve 
curve.columns = ['k_value', 'rmse']
k_array = np.asarray(k_list)
rmse_val_array = np.asarray(rmse_val)

fig = go.Figure(data = go.Scatter(x= curve['k_value'], y=curve['rmse']))
fig.update_layout(
        title = 'Accuracy per number of K Neighbors',
        xaxis = dict(
                tick0 = 3,
                dtick = 2,
                title_text = '# of Neighbors'
                ),
        yaxis = dict(
                title_text = 'Root Mean Squared Error')
            )
        
plot(fig)



    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#    model = sm.OLS(y_train, X_train).fit()
#    print_model = model.summary()    
    knn_model = sm.OLS(y_train, X_train).fit()
    knn_model.summary()
    pVals = knn_model.pvalues
    pVals = pVals.to_frame().reset_index(drop = False)
    pVals.columns= ['index', 'pvalue']








test = list(range(0,8))


variables_2 = ['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday']
ride_type_2 = ['Standard'] 
cab_type_2 = ['Uber', 'Lyft']



curve.iloc[:,1]

































def reg_params(variables, cab_type, ride_type):
    df_filter = df.loc[(df['ride_type'].isin(ride_type)) & (df['cab_type'].isin(cab_type))]
#    X = df_filter.filter(items = variables) 
    X = df_filter[variables]    
    Y = df_filter['price']
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    model = sm.OLS(y_train, X_train).fit()
#    print_model = model.summary()
    return model.summary()
#    return model.pvalues




reg_params(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], 'Uber', 'Shared')

##remove humidity
reg_params(['distance', 'temperature','precipIntensity', 'windSpeed', 'time', 'weekday'], 'Uber', 'Shared')

##remove precipIntensity
reg_params(['distance', 'temperature', 'windSpeed', 'time','weekday'], 'Uber', 'Shared')

##remove time
reg_params(['distance', 'temperature', 'windSpeed', 'weekday'], 'Uber', 'Shared')

##remove windspeed
reg_params(['distance', 'temperature', 'weekday'], 'Uber', 'Shared')

##remove temperature
reg_params(['distance', 'weekday'], ['Uber', 'Lyft'], ['Shared'])






sigLevel = 0.1
pVals = reg_params(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber'], ['Shared'])


while max(pVals)>0.05:
    for x,y in zip(pVals,xtrain.columns): 
        if x>0.05:
            xtrain = xtrain.drop(y,axis=1)
            xtest = xtest.drop(y,axis=1)
            lrmodel = sm.OLS(ytrain,xtrain).fit()
            break
# after all the values are less than 0.05, assign the model to final model
finalmodel = lrmodel



variables = ['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday']

df_filter = df.loc[(df['ride_type'].isin(['Shared'])) & (df['cab_type'].isin(['Uber']))]
#    X = df_filter.filter(items = variables) 
X = df_filter[variables]
Y = df_filter['price']
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
lrmodel = sm.OLS(y_train, X_train).fit()
#    print_model = model.summary()
sigLevel = 0.1    

lrmodel = sm.OLS(y_train, X_train).fit()
pVals = lrmodel.pvalues
pVals = pVals.to_frame().reset_index(drop = False)
pVals.columns= ['index', 'pvalue']
while max(pVals['pvalue'])>0.05:
    for p in range(0, len(pVals['pvalue'])): 
        p_value = pVals['pvalue'][p]
        i_max = pVals['pvalue'].values.argmax()
        column_drop = pVals['index'][i_max]
    X_train.drop([column_drop], axis = 1, inplace = True)
    X_test.drop([column_drop], axis = 1, inplace = True)
    lrmodel = sm.OLS(y_train,X_train).fit()
    pVals= lrmodel.pvalues
    pVals = pVals.to_frame().reset_index(drop = False)
    pVals.columns= ['index', 'pvalue']

lrmodel = sm.OLS(y_train, X_train).fit()
lrmodel.summary()




pVals.summary()

maxd = pVals.iloc[pVals['pvalue'].idxmax]

pVals['pvalue'].values.argmax()



while max(pVals)>0.05:
    for x in enumerate(lrmodel.pvalues): 
        print(max(lrmodel.pvalues))




        
        if x>0.05:
            X_train= X_train.drop(y,axis=1)
            X_test = X_test.drop(y,axis=1)
            lrmodel = sm.OLS(y_train,X_train).fit()
            break
finalmodel = lrmodel
finalmodel.summary()

X_train.drop(np.argmax(pVals), axis = 1)

max(pVals)


z = zip(pVals,X_train.columns)
print(z)

























