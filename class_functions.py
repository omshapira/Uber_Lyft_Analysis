#https://towardsdatascience.com/object-oriented-programming-for-data-scientists-build-your-ml-estimator-7da416751f64
#https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/OOP_in_ML/Class_MyLinearRegression.py


%reset


from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm
from scipy.stats import f as fisher_f
from sklearn import metrics
from scipy import stats
from sklearn.linear_model import LinearRegression 
import numpy as np

class MyLinearRegression:    
    def __init__(self, features, cab_type, ride_type): #build instance
        self.features= features
        self.cab_type = cab_type
        self.ride_type = ride_type     
    def filter_data(self):
        self.features = self.features
#        features = self.features.insert(0, 'price')
#        global df
        df_filter = df.loc[(df['ride_type'].isin(self.ride_type)) & (df['cab_type'].isin(self.cab_type))]
        df_filter = df_filter.filter(items = self.features)
        if any('weekday' in s for s in df_filter.columns):
            df_encode = df_filter['weekday']
            df_filter = pd.concat([df_filter, pd.get_dummies(df_encode, prefix = 'Day', drop_first = True)], axis = 1)
            df_filter.drop(['weekday'], axis=1, inplace = True)
            df_filter = df_filter.reset_index(drop = True)
        else:
            df_filter = df_filter.reset_index(drop = True)                
        return df_filter

    def split_data(self):
        df_filter = self.filter_data()
        X = df_filter.loc[:, df_filter.columns != 'price']
        Y = df_filter['price']
    #    X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        return X_train, X_test, y_train, y_test

    def create_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        model_cab = LinearRegression()
#        model_cab.fit(X_train, y_train)
        return model_cab

    def predict_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        return_model = self.create_model()
        fit_model = return_model.fit(X_train, y_train)
        predict = return_model.predict(X_test)
        return predict

    def linear_formula(self):
        X_train, X_test, y_train, y_test = self.split_data()
        return_model = self.create_model()
        fit_model = return_model.fit(X_train, y_train)
        slope = fit_model.coef_[0]
        intercept = fit_model.intercept_
        print(f"The linear regression line for " + "is y = " + str(round(slope,2))+ "x + " + str(round(intercept,2))) 

    def full_linear_formula(self):
        X_train, X_test, y_train, y_test = self.split_data()
        return_model = self.create_model()
        fit_model = return_model.fit(X_train, y_train)
        slope = fit_model.coef_
        intercept = fit_model.intercept_
        return slope
#        print(f"The linear regression line for " + "is y = " + str(round(slope,2))+ "x + " + str(round(intercept,2))) 

        
    def metric_r2(self):
        X_train, X_test, y_train, y_test = self.split_data()
        return_model = self.create_model()
        fit_model = return_model.fit(X_train, y_train)
        r2_score = return_model.score(X_test, y_test)
        return round(r2_score,3)

    def metric_MAE(self):
        predicted = self.predict_model()        
        model_cab = self.create_model()
        X_train, X_test, y_train, y_test = self.split_data()
        model_cab.fit(X_train, y_train)
        MAE= metrics.mean_absolute_error(predicted, y_test)
        return round(MAE,3)

    def metric_MSE(self):
        predicted = self.predict_model()        
        model_cab = self.create_model()
        X_train, X_test, y_train, y_test = self.split_data()
        model_cab.fit(X_train, y_train)
        MAE= metrics.mean_squared_error(predicted, y_test)
        return round(MAE,3)

    def metric_RMSE(self):
        predicted = self.predict_model()        
        model_cab = self.create_model()
        X_train, X_test, y_train, y_test = self.split_data()
        model_cab.fit(X_train, y_train)
        MAE= np.sqrt(metrics.mean_squared_error(predicted, y_test))
        return round(MAE,3)

    
#    MAE_all = metrics.mean_absolute_error(predict_all, y_test_all)
#    MSE_all = metrics.mean_squared_error(predict_all, y_test_all)
#    RMSE_all = np.sqrt(metrics.mean_squared_error(predict_all, y_test_all))
#

all_lin_reg_multi = MyLinearRegression(['price','distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'],['Standard'])

test = all_lin_reg_multi.filter_data()
all_lin_reg_multi.linear_formula()



all_lin_reg_multi.metric_MSE()


all_lin_reg_simple.filter_data()

all_lin_reg_simple.full_linear_formula()







###########chow test

from scipy.stats import f as fisher_f
def chow_test(alpha_val, ride_type):

    ##get dataset for both Uber + Lyft
    all_lin_reg_simple = MyLinearRegression(['price','distance', ], ['Uber', 'Lyft'],['Standard'])
    y_test_all = all_lin_reg_simple.predict_model()
    predict_all = np.asarray(all_lin_reg_simple.split_data()[3])
    
    ##Predictions for Uber Only
    uber_lin_reg_simple = MyLinearRegression(['price','distance', ], ['Uber'],['Standard'])
    y_test_uber = uber_lin_reg_simple.predict_model()
    predict_uber = np.asarray(uber_lin_reg_simple.split_data()[3])
    ##Predictions for Lyft Only
    lyft_lin_reg_simple = MyLinearRegression(['price','distance', ], ['Lyft'],['Standard'])
    y_test_lyft = lyft_lin_reg_simple.predict_model()
    predict_lyft = np.asarray(lyft_lin_reg_simple.split_data()[3])
    
    ##Add up Sum of Squared Errors
    SSE_all = sum((np.asarray(y_test_all - predict_all))*(np.asarray(y_test_all - predict_all)))
    SSE_lyft = sum((np.asarray(y_test_lyft - predict_lyft))*(np.asarray(y_test_lyft - predict_lyft)))
    SSE_uber = sum((np.asarray(y_test_uber - predict_uber))*(np.asarray(y_test_uber - predict_uber)))

    N = len(y_test_all)
    deg_freedom_all = len(y_test_all) - 1 
    deg_freedom_1_lyft = len(y_test_lyft) - 1
    deg_freedom_2_uber = len(y_test_uber) - 1
    
    k = 2 # one dimensional regression - slope and intercept
    numerator = (SSE_all - (SSE_lyft + SSE_uber))/k
    denominator = (SSE_lyft + SSE_uber)/(N - 2*k) #both regression have 2 degrees of freedom
    
    f_statistics = numerator/denominator #calculate f-statistic significant value for each months' regressions
    
    alpha = alpha_val
    p_value = fisher_f.cdf(f_statistics, 2, N-2*2) ##calculate critical value for degrees 2 and 2
    
    if p_value > alpha:
        chow_test = str('The Chow test shows that there is a statistically significant difference between the Uber and Lyft regression lines')
    else:
        chow_test = str('The Chow test shows that there is not a statistically significant difference between the Uber and Lyft regression lines')
    
    
    return chow_test

chow_test(.05,['Standard'])














































    uber_lin_reg_multi = MyLinearRegression(['distance', 'temperature','precipIntensity', 'humidity', 'windSpeed', 'time', 'weekday'], ['Uber', 'Lyft'],['Standard'])
uber_lin_reg_multi.filter_data()

    features = ['price', 'distance', 'ride_type', 'cab_type']
    df_filter = df.filter(items = features) 
    df_filter_all = df_filter.loc[(df_filter['cab_type'].isin(cab_type) & (df_filter['ride_type'].isin(ride_type)))]
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





def df_regression_filter(variables, cab_type, ride_type):
    df_filter = df.loc[(df['ride_type'].isin(ride_type)) & (df['cab_type'].isin(cab_type))]
    df_filter = df_filter.filter(items = variables)   
    return df_filter






class Employee_2:
    
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): #constructor/initializor with arguments we want to accept
        self.fname= first ##could also match name with arugments like self.first
        self.lname = last
        self.pay = pay
        self.email = first+ '.' + last+ '@company.com'
    def full(self):
        return self.lname

    def raise_func(self):
        return self.pay


emp_1 = Employee_2('Corey', 'Schafer', 50000)


print(emp_1.raise_func())

print(emp_1.full())

print(Employee_2.__dict__)





class Employee:
    
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): #constructor/initializor with arguments we want to accept
        self.fname= first ##could also match name with arugments like self.first
        self.lname = last
        self.pay = pay
        self.email = first+ '.' + last+ '@company.com'
        
    def fullname(self): ##create method within class
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
            #for raise amt, can't just enter 'raise_amount'. Class variable must be accessed within class itself or instance. Either below works
#        self.pay = int(self.pay * self.raise_amount) 
        self.pay = int(self.pay * Employee.raise_amount) #can't just enter 'raise_amount'. Class variable must be accessed within class itself or instance
        return self.pay
    
    def test_func(self):
        return self.email

emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('Test', 'User', 60000)

print(emp_1.test_func())


print(emp_1.email)
print(emp_2.email)
print(emp_1.fullname()) #here, don't hve to pass the instance

print(emp_1.pay)


Employee.raise_amount = 1.05 ##changes for the entire class and all it's instances
emp_1.raise_amount = 1.05 #changes raise amount for only employee_1

print(emp_1.apply_raise())
print(Employee.raise_amount)
print(emp_1.raise_amount) ##the instance emp_1 is inheritiing raise_amount from the class

print(emp_1.__dict__) ##contains all info for emp_1 instance. Raise_amount does NOT appear
print(Employee.__dict__)

Employee.fullname(emp_1) #here, instance of emp_1 has to be specified


    pass


emp_1 = Employee()
emp_2 = Employee() ##both of these are instances of class with different locations in memory

emp_1.first = 'Corey'
emp_1.last= 'Schafer'
emp_1.email = 'Corey.Schafer@company.com'
emp_1.pay = 50000

emp_2.first = 'Test'
emp_2.last= 'User'
emp_2.email = 'Test.User@company.com'
emp_2.pay = 60000








class Polygon:
    def __init__(self, no_of_sides):
        self.n = no_of_sides
        self.sides = [0 for i in range(no_of_sides)]
    def inputSides(self):
        self.sides = [float(input("Enter side "+str(i+1)+" : ")) for i in range(self.n)]
    def dispSides(self):
        for i in range(self.n):
            print("Side",i+1,"is",self.sides[i])

















class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

class Square:
    def __init__(self, length):
        self.length = length

    def area(self):
        return self.length * self.length

    def perimeter(self):
        return 4 * self.length