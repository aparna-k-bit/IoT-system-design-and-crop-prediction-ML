# -*- coding: utf-8 -*-


# Support Vector Machine (SVM)

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
READ_API_KEY='put own key here'
CHANNEL_ID= 'put own key here'

"""## Importing the dataset"""

data = pd.read_csv('cpdata.csv')


X = data.iloc[:, 0:4].values
y = data.iloc[:, 4:].values

"""## Splitting the dataset into the Training set and Test set"""

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

"""## Training the SVM model on the Training set"""

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train.ravel())




"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))




from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a=accuracy_score(y_test,pred)
print("The accuracy of this model is: ", a*100)


# Commented out IPython magic to ensure Python compatibility.
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
l=[]
l.append(atemp)
l.append(ah)
l.append(pH)
l.append(rain)
predictcrop=[l]

# Putting the names of crop in a single list
crops=['rice','wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']

predictions = classifier.predict(predictcrop)
print(predictions)



