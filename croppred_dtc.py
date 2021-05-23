
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

READ_API_KEY='put own key'
CHANNEL_ID= 'put your own key here'

#Reading the csv file
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
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()

#Fitting the classifier into training set
clf.fit(X_train,y_train)
pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a=accuracy_score(y_test,pred)
print("The accuracy of this model is: ", a*100)
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
crops=['wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']
cr='rice'

predictions = clf.predict(predictcrop)
count=0
for i in range(0,30):
    if(predictions[0][i]==1):
        c=crops[i]
        count=count+1
        break;
    i=i+1
if(count==0):
    print('The predicted crop is %s'%cr)
else:
    print('The predicted crop is %s'%c)

