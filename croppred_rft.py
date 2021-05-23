# -*- coding: utf-8 -*-
#Ramdom Forest Regressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
READ_API_KEY='put own key here'
CHANNEL_ID= 'put own key here'
from sklearn.model_selection import train_test_split

data= pd.read_csv('cpdata.csv')

data=pd.read_csv('cpdata.csv')
print(data.head(1))

#Creating dummy variable for target i.e label
label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
print('The data present in one row of the dataset is')
print(data.head(1))
train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values

#Dividing the data into training and test set
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.8)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

plt.matshow(data.corr())

import seaborn as sns
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

def low_cardinality_cols(data_frame):
    low_cardinality_cols = [cname for cname in data_frame.columns if 
                                data_frame[cname].nunique() < 70 and
                                data_frame[cname].dtype == "object"]
    return(low_cardinality_cols)
  
cat_data_features =low_cardinality_cols(data)

# Identify numeric columns
def numeric_cols(data_frame):
    numeric_cols = [cname for cname in data_frame.columns if 
                                data_frame[cname].dtype in ['int64', 'float64']]
    return(numeric_cols)
  
  
num_data_features = numeric_cols(data)

np.nan_to_num(X_train)
np.where(np.isnan(X_train))
X_train=np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth = 6 , n_estimators = 30 ,min_samples_split = 2)

np.where(np.isnan(X_train))

model.fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error
Train_accuracy = model.score(X_train,y_train)
Test_accuracy = model.score(X_test,y_test)
print('Train_accuracy: ',Train_accuracy)
print('Test accuracy: ',Test_accuracy)

# Commented out IPython magic to ensure Python compatibility.
TS = urllib.request.urlopen("http://api.thingspeak.com/channels/%s/feeds/last.json?api_key=%s" \
#                        % (CHANNEL_ID,READ_API_KEY))
TS1 = urllib.request.urlopen("http://api.thingspeak.com/channels/%s/feeds/14.json?api_key=%s" \
#                        % (CHANNEL_ID,READ_API_KEY))
TS2 = urllib.request.urlopen("http://api.thingspeak.com/channels/%s/feeds/13.json?api_key=%s" \
#                        % (CHANNEL_ID,READ_API_KEY))
TS3 = urllib.request.urlopen("http://api.thingspeak.com/channels/%s/feeds/12.json?api_key=%s" \
#                        % (CHANNEL_ID,READ_API_KEY))

response = TS.read()
response1 = TS1.read()
response2 = TS2.read()
response3 = TS3.read()

data = json.loads(response)
data1 = json.loads(response1)
data2 = json.loads(response2)
data3 = json.loads(response3)

a = data['created_at']
atemp = data['field1']
ah= data1['field2']
pH= data2['field3']
rain= data3['field4']


l=[]
l.append(atemp)
l.append(ah)
l.append(pH)
l.append(rain)
predictcrop=[l]

# Putting the names of crop in a single list
crops=['rice','wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']

#Predicting the crop
predictions = model.predict(predictcrop)
count=0
print(predictions)



maxxx=np.amax(predictions)

count=0
for i in range(0,30):
    if (predictions[0][i]==maxxx):
        c=i;
        break
print(crops[i])

