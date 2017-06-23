'''
Regression
    -predicting house price
'''


##fire up graphlab create
import graphlab

##load house sales data
sales = graphlab.SFrame('home_data.gl/')
#print sales

##exploring the data for housing sales (do some visualization)
#sales.show(view='Scatter Plot', x='sqft_living', y='price')

##create a simple regression model of sqft_living to price
##firstly, split traning dataset and test dataset randomly 
##arguments: 0.8mieans 80% for traning; set seed to make it split in the same way each time)
train_data, test_data = sales.random_split(.8,seed=0)

##build the regression model
##arguments: dataset; target is the value to be predicted (y-axis); feature is a list of factors that prediction relied on(x-axis)
sqft_model = graphlab.linear_regression.create(train_data, target='price',
                                               features=['sqft_living'])

##evluate error(RMSE) of the simple model
#print test_data['price'].mean()
#print sqft_model.evaluate(test_data)

##show coeffients
#print sqft_model.get('coefficients') 

##show what our predicitons look like
#import matplotlib.pyplot as plt
#plt.plot(test_data['sqft_living'],test_data['price'],'.',
#         test_data['sqft_living'],sqft_model.predict(test_data),'-')


##explore other features in the data
my_features = ['bedrooms','bathrooms', 'sqft_living', 'sqft_lot','floors', 'zipcode']
#sales[my_features].show()
#sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

##build a regression model with more featuires
my_features_model = graphlab.linear_regression.create(train_data, target='price',
                                                      features=my_features)
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)


##apply learned models to predict prices of specific houses
#house1 = sales[sales['id']=='5309101200']
#print house1['price']
#print sqft_model.predict(house1)
#print my_features_model.predict(house1)
#house2 = sales[sales['id']=='1925069082']
#print house2['price']
#print sqft_model.predict(house2)
#print my_features_model.predict(house2)   
bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}
print my_features_model.predict(graphlab.SFrame(bill_gates))


'''
##quiz working
import graphlab
sales = graphlab.SFrame('home_data.gl/')
#print sales[sales['zipcode'] == '98039']['price'].mean()
#selected = len(sales[(sales['sqft_living'] < 4000) and (sales['sqft_living'] > 2000)])
#print float(selected) / len(sales)
train_data, test_data = sales.random_split(.8,seed=0)
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
my_features_model = graphlab.linear_regression.create(train_data, target='price',
                                                      features=my_features,
                                                      validation_set=None)
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]
advanced_features_model = graphlab.linear_regression.create(train_data, target='price',
                                                            features=advanced_features,
                                                            validation_set=None)
print my_features_model.evaluate(test_data)
print advanced_features_model.evaluate(test_data)
'''





                            