#!/usr/bin/env python
# coding: utf-8

# ## <center> $\textbf{Flat Price Prediction}$ 

# In[246]:


# I have used the dataset("resale-flat-prices-based-on-approval-date-1990-1999.csv") 
# for the flat price prediction by considering the number of rooms 
# according to the rules of Singapore Housing and Development Board (HDB). 

# For more details: https://www.hdb.gov.sg/residential/buying-a-flat/resale/getting-started/types-of-flats
# Datasets: https://data.gov.sg/dataset/resale-flat-prices?resource_id=42ff9cfe-abe5-4b54-beda-c88f9bb438ee


# In[247]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[248]:


#reading .csv file and storing it in "dataset" dataframe
dataset=pd.read_csv("/home/ankit/Desktop/resale-flat-prices-based-on-approval-date-1990-1999.csv")


# In[249]:


print(dataset)
# There are total 9 features(predictors) including both qualitative and quantitative features.


# In[250]:


print(dataset.shape)
# Total 287200 rows and 10 columns are present in this dataset.


# In[251]:


dataset.flat_type.unique() # to find the unique values in flat_type column


# In[252]:


n1 = len(pd.unique(dataset['flat_type']))
n1


# In[253]:


dataset.storey_range.unique() # to find the unique values in storey_range column


# In[254]:


dataset.describe()
# We get the statistic about the dataset.
# On average, floor area is 93.35 sqm and resale price is 219541$ and we can also check the other details here.
# From this output, I will try to see the nature of the dataset.


# In[255]:


#summary statistics of columns in dataframe
dataset.describe().transpose() # Rowwise summary


# In[256]:


dataset.dtypes # To see the datatypes of the column data


# ## $\textbf{Data Preprocessing}$

# In[257]:


data = dataset.to_numpy() # storing dataset as numpy array


# In[258]:


#dataset['flat_type'] = dataset['flat_type'].map(lambda x: x.rstrip('ROOM'))


# In[259]:


# https://www.hdb.gov.sg/residential/buying-a-flat/resale/getting-started/types-of-flats
# Converting the string type flat_type data into numerical data, so that machine learning algorithm
# will work on this numerical data.
# Used the number of bedrooms in each flat_type according to the rules of HDB(singapore) and assumed number 
# of bedrooms in "1 ROOM" flat_type is 1 as it is not mentioned on the site.
for i in range(len(data)):
    if data[i][2]=="1 ROOM":
        data[i][2]=1
    if data[i][2]=="2 ROOM":
        data[i][2]=1
    if data[i][2]=="3 ROOM":
        data[i][2]=2
    if data[i][2]=="4 ROOM":
        data[i][2]=3
    if data[i][2]=="5 ROOM":
        data[i][2]=3
    if data[i][2]=="EXECUTIVE":
        data[i][2]=3
    if data[i][2]=="MULTI GENERATION":
        data[i][2]=4    


# In[260]:


# In the storey_range column, on taking the average of minimum and maximum number of storeys, so that 
# string get converted into numerical data. 
# Extracted first 2 digits and last 2 digits and have taken the average.
for i in range(len(data)):
    string=data[i][5]
    p=int(string[0])
    q=int(string[1])
    x=10*p + q
    #print(x)
    r=int(string[-1])
    s=int(string[-2])
    y=10*s+r
    #print((x+y)/2)
    avg=round((x+y)/2)
    data[i][5]=avg


# In[261]:


# From numpy array i.e. "data"(list of lists) to dataframe:
dataset1 = pd.DataFrame(data, columns =['month','town','flat_type','block','street_name','storey_range','floor_area_sqm','flat_model','lease_commence_date','resale_price']) 


# In[262]:


dataset1


# ## $\textbf{Exploratory Data Analysis}$

# In[263]:


dataset['flat_type'].value_counts().plot(kind='bar')
plt.title('Number of bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine()


# In[264]:


# As we can see from the visualization 3 and 4 bedroom houses are most commonly sold.
# So, for a builder having this data , it can make a new flat with more 3 and 4 bedrooms
# to attract more buyers.


# In[265]:


dataset['storey_range'].value_counts().plot(kind='bar')
plt.title('Number of storeys_ranges')
plt.xlabel('Storeys')
plt.ylabel('Count')
sns.despine()


# In[266]:


# As we can see 4 to 6, 7 to 9 , 1 to 3 and 10 to 12 storey_range flats have more count.
# So, to predict resale flat prices of the flat, we should have to consider these storey range flats.


# In[267]:


dataset['flat_model'].value_counts().plot(kind='bar')
plt.title('Flat_Model')
plt.xlabel('flat_models')
plt.ylabel('Count')
sns.despine()


# In[268]:


# Here, "NEW GENERATION", "IMPROVED" and "MODEL A" flat models have more count compared to other flat models.
# So, while predicting flat prices, we should have to concentrate these flat models.


# In[269]:


sns.pairplot(dataset,hue='flat_model', diag_kws={'bw': 0.2})
plt.show()


# In[270]:


# As we can see, there is a spike in the scatter plot between floor_area_sqm and lease commence_date.
# For lease_commence year between 1960 and 1980, for 2-ROOM flat model, floor area is very high and this 
# 2-ROOM flat model is used between these years only.
# Similarly, APARTMENT,MODEL A-MAISONETTE and MAISONETTE flat models were used between 1980 and 2000
# lease_commence year.
# For "MODEL A" AND "STANDARD" flat models, flat area is under 150 square meters.
# There is approximate a linear relationship between resale_price and floor_area_sqm
# MODEL A and STANDARD flat models have floor_area between 100 and 200 sqm for which resale price is between 
# 200000 and 700000
# PREMIUM APARTMENT have floor_area between 0 and 150 sqm for which resale_prices are under 600000$.
# For "most" 2-ROOM and TERRACE flat models, lease_commence year is between 1980 and 1998 and 
# resale_prices are above 400000$.


# In[271]:


sns.pairplot(dataset,hue='flat_type', diag_kws={'bw': 0.2})
plt.show()


# In[272]:


# There is a spike for "3-ROOM" flat_type for earlier lease_commence_year of 1980 which shows 
# total flat area in sqm is maximum in that year for "3-ROOM" flat type
# floor area for "4 ROOM" and "5 ROOM" flat_type is less than 150 sqm.
# floor area for "1 ROOM" and "2 ROOM" flat_type is less than 50 sqm.
# EXECUTIVE flat type are implemented after 1980 lease commence_year and mostly floor are is between 
# 150 and 200 sqm except 2-3 flats.
# Mostly EXECUTIVE flats have resale price more than 400000$ with floor area around 100 sqm.
# "4 ROOM" flats have resale price less than 600000$


# In[273]:


sns.pairplot(dataset,hue='storey_range', diag_kws={'bw': 0.2})
plt.show()


# In[274]:


dataset


# In[275]:


dataset.corr() # Finding Correlation between features:


# In[276]:


sns.heatmap(dataset.corr(),annot=True,lw=1)


# In[277]:


# Correlation is a statistical measure to explain the relationship between two or more than 
# two variables which are used to predict the values of target variable.
# If two variables or features are positively correlated with each other,
# it means when the value of one variable increases then the value of the other variable(s) also increases. 


# In[278]:


# resale_price and floor_area_sqm are highly correlated


# In[279]:


# Box-plots (another way of visualizing and analysing data with min,max,25,50 and 75 percentile values)
sns.boxplot(y='resale_price',x='flat_type',data=dataset1)


# In[280]:


sns.boxplot(y='resale_price',x='flat_model',data=dataset1)


# In[281]:


sns.boxplot(y='resale_price',x='storey_range',data=dataset1)


# In[282]:


import matplotlib.pyplot as plt
plt.hist(dataset.resale_price)
plt.show()


# In[283]:


# flats which have resale prices are between 100000$ and 200000$ have highest count which is 80000
# and then comes those flats which have resale prices 200000$ and 300000$ with 70000 count
# and then flats with resale prices between 0 and 100000$ comes with count ~50000


# In[284]:


colors = np.where(dataset.resale_price > 1, 'r', 'k')
#plt.scatter(dataset.resale_price, dataset.floor_area_sqm, s=20, c=colors)
# OR (with pandas 0.13 and up)
dataset.plot(kind='scatter', x='floor_area_sqm', y='resale_price', s=20, c=colors)


# In[285]:


import seaborn as sns
sns.scatterplot(x="storey_range", y="resale_price", data=dataset)


# In[286]:


import seaborn as sns
sns.scatterplot(x="flat_model", y="resale_price", data=dataset)


# In[287]:


sns.scatterplot(x="flat_type", y="resale_price", data=dataset)


# In[288]:


# Based on parameters "floor_area_sqm", "flat_type","flat_model","storey_range" as above, we can see 
# the range of resale flat prices.


# ### $\text{On Adding latitude and longitude columns based on the address given in dataset}$

# In[289]:


# As the record size is 287200 in the given dataset and there is a problem of time out if we use geopy 
# library to convert given address into latitude and longitude. So, just to analyze the data, I am considring
# less record size with 1000 rows.


# In[290]:


modified_dataset=dataset=pd.read_csv("/home/ankit/Desktop/resale-flat-prices-based-on-approval-date-1990-1999.csv", nrows = 1000)


# In[291]:


modified_dataset.shape


# In[292]:


dataset["address"] = dataset["town"] + " " + dataset["block"] + " " + dataset["street_name"] 


# In[293]:


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
import time
Address_info= dataset[['town','block','street_name']].copy()
Address_info = Address_info.apply(lambda x: x.str.strip(), axis=1)  
Address_info['Full_Address'] = Address_info[Address_info.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

locator = Nominatim(user_agent="myGeocoder")  
def geocode_me(location):
    time.sleep(1.1)
    try:
        return locator.geocode(location)
    except (GeocoderTimedOut, GeocoderQuotaExceeded) as e:
        if GeocoderQuotaExceeded:
            print(e)
        else:
            print(f'Location not found: {e}')
            return None

Address_info['location'] = Address_info['Full_Address'].apply(lambda x: geocode_me(x)) 
Address_info['point'] = Address_info['location'].apply(lambda loc: tuple(loc.point) if loc else None)
Address_info[['latitude', 'longitude', 'altitude']] =   pd.DataFrame(Address_info['point'].tolist(), index=Address_info.index)
#Ref:Stackoverflow


# In[294]:


modified_dataset=pd.concat([modified_dataset,Address_info], axis=1)


# In[295]:


modified_dataset


# In[296]:


new_dataset=modified_dataset.dropna() #Removing where values are undefined in the form of 'NaN'


# In[297]:


new_dataset


# ### $\text{Now, We are going to see the common locations where the flats are placed.}$

# In[298]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distributions of latitude and longitude in the Resale Flat Prices dataset', fontsize=16)
ax1.hist(new_dataset.latitude)
ax1.set_xlabel('Latitude', fontsize=13)
ax1.set_ylabel('Frequency', fontsize=13)
ax2.hist(new_dataset.longitude)
ax2.set_xlabel('Longitude', fontsize=13)
ax2.set_ylabel('Frequency', fontsize=13);


# In[299]:


# As we can see with latitude range 0 to 10, maximum flats were sold and same for longitude around 100.
# So, these locations might be ideal location for flat sale in future also.


# In[300]:


from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
geometry = [Point(xy) for xy in zip(new_dataset['longitude'], new_dataset['latitude'])]
gdf = GeoDataFrame(new_dataset, geometry=geometry)   
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);
#Ref: Stackoverflow


# In[301]:


import geopandas
gdf = geopandas.GeoDataFrame(new_dataset, geometry=geopandas.points_from_xy(new_dataset.longitude, new_dataset.latitude))


# In[302]:


print(gdf.head())


# In[303]:


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# We restrict to Singapore.
ax = world[world.continent == 'Singapore'].plot(color='white', edgecolor='black')
gdf.plot(ax=ax, color='red')

plt.show()


# In[304]:


plt.figure(figsize = (10,8))
plt.scatter(new_dataset.longitude, new_dataset.latitude ,c=new_dataset.resale_price, cmap = 'cool', s=1)
plt.colorbar().set_label('Resale Flat Price ($)', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.title('Resale Flat Price ($)', fontsize=17)
plt.show()


# In[2]:


# As we can see Locations for which longitude is between -100 to 50 and latitude between 30 to 50,
# Flat prices are below 5000$.


# ## $\textbf{Building the Model: Predicting the resale flat price}$

# In[306]:


## Once we get a good fit, we will use this model to predict the sale price of the flat.


# In[307]:


#modified_dataset=pd.concat([modified_dataset,Address_info], axis=1)


# In[308]:


#modified_dataset


# In[309]:


#array_train_data=train_dataset.to_numpy()


# In[310]:


columns_to_remove = (0,1,3,4,7,8)
new_object = [[x for i,x in enumerate(l) if i not in columns_to_remove] for l in data]


# In[1]:


#new_object


# In[312]:


index_values = [i for i in range(len(data))] 
   
# creating a list of column names 
column_values = ['flat_type','floor_area_sqm', 'number_of_storeys', 'resale_price'] 
  
# creating the dataframe 
df = pd.DataFrame(data = new_object,  
                  index = index_values,  
                  columns = column_values) 
  
# displaying the dataframe 
#print(df) 


# In[313]:


i = list(df.columns)
a, b = i.index('floor_area_sqm'), i.index('number_of_storeys')
i[b], i[a] = i[a], i[b]
df = df[i]


# In[314]:


df = df.rename(columns={'floor_area_sqm': 'number_of_storeys','number_of_storeys': 'floor_area_sqm'})


# In[315]:


df


# # $\textbf{Implementation from scratch:}$
# 
# ## $\text{Equation of Best Fit Hyperplane: $z^* = (A^TA)^{-1} A^TB$ for system of equations $Az=b$}$ 
# ## $\text{where vector $b$ is not in plane of column vectors of matrix $A$ and to get the}$ 
# ## $\text{approximate solution, we have projected the vector $b$ in plane which is $\hat{b}$}$
# ## $\text{and so solving $Az^* = \hat{b},$ we get $z^* = (A^TA)^{-1} A^TB$}$

# In[316]:


dataset=df.to_numpy() #converting dataframe to numpy array


# In[317]:


# writing it in z= ax+by+c form and then convert it into matrix form
# I am writing it in matrix equation form directly

A=[]
for i in range(10):
    temp=[]
    temp.append(dataset[i][0])
    temp.append(dataset[i][1])
    temp.append(1) # for coefficient of c i.e. 1
    A.append(temp)

B=[]
for i in range(10):
    B.append(dataset[i][-1])    


# In[318]:


# Printing matrix A and vector B
print(A)
print(B) 


# In[319]:


#writing function for transpose of a matrix
def transpose(A,m,n):
    trans=[]
    for j in range(n):
        temp=[]
        for i in range(m):
            temp.append(A[i][j])
        trans.append(temp)
    return trans    


# In[320]:


trans_A= transpose(A,10,3)
print(transpose(A,10,3)) # printing transpose of matrix A


# In[321]:


mul = np.dot(trans_A, A) # multiplying A^T and A
inv = np.linalg.inv(mul) # inverse of A^T*A
prod= np.dot(inv,trans_A) # multiplying (A^T*A)^-1 and A^T
res = np.dot(prod,B) # finding (A^T*A)^-1 * A^T* B


# In[322]:


# So, our result matrix is 
print(res) # It shows the values of a,b,c


# In[323]:


### Conclusion : Best fit hyperplane for the given data is z = 7750*x + 766.666*y - 24266.66
#### To estimate the resale_price, I will use this best fit hyperplane  
### ----------------------------------------------------------------------------------------------------------


# In[ ]:





# ## $\textbf{Using already implemented Linear Regression}$

# In[324]:


#df.dropna()     #drop all rows that have any NaN values


# In[325]:


X = df[['flat_type','floor_area_sqm','number_of_storeys']]


# In[326]:


X.head()


# In[327]:


Y = df['resale_price']


# In[328]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[329]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[330]:


# print the intercept
print(model.intercept_)


# In[331]:


coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_parameter
#The sign of each coefficient indicates the direction of the relationship
#between a predictor variable and the response variable.
# A positive sign indicates that as the predictor variable increases, the Target variable also increases.
#A negative sign indicates that as the predictor variable increases, the Target variable decreases.


# In[332]:


predictions = model.predict(X_test)
predictions


# In[333]:


sns.regplot(y_test,predictions)


# In[334]:


model.score(X_test,y_test)


# In[335]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,learning_rate = 0.1, loss = 'ls')


# In[336]:


clf.fit(X_train, y_train)
clf.score(X_test,y_test)


# In[337]:


import statsmodels.api as sm
X_train_Sm= sm.add_constant(X_train)
#X_train_Sm= sm.add_constant(X_train)
ls=sm.OLS(y_train,X_train_Sm).fit()
print(ls.summary())


# # $\textbf{Conclusion}$

# ## $\text{Many factors are affecting the resale prices of the flat, like floor_area 
# which increases}$<br> $\text{the price of the flat and even location of the flat influencing the prices of the flat.}$

# In[ ]:




