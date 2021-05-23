

# K-Nearest Neighbors (K-NN)

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
READ_API_KEY='put own key'
CHANNEL_ID= 'put your own key here'

"""## Importing the dataset"""

data=pd.read_csv('cpdata.csv')
print(data.head(1))

#Creating dummy variable for target i.e label
label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
print('The data present in one row of the dataset is')
print(data.head(1))
X=data.iloc[:, 0:4].values
y=data.iloc[: ,4:].values

"""
## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(X_train)

print(y_train)

print(X_test)

print(y_test)

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

print(X_test)

"""## Training the K-NN model on the Training set"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score

accuracy_score(y_test, y_pred)


#atemp = 33
#ah= 24
#pH= 6
#rain= 200
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

crops=['wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']
cr='rice'

#Predicting the crop


predictions = classifier.predict(predictcrop)

predictions

