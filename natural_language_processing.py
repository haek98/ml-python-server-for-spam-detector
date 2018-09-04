# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import socket   
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
import re
import nltk
import struct
from sklearn.linear_model import LogisticRegression
import copy
# Importing the dataset
dataset = pd.read_csv('spam.csv',encoding='latin1')
drop=['Unnamed: 2','Unnamed: 3','Unnamed: 4']
dataset.drop(drop,axis=1,inplace=True)
# Cleaning the texts
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, dataset.shape[0]):
    message = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(message)
    corpus.append(message)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=6000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:dataset.shape[0], 0:1].values
labelencoder = LabelEncoder()
y[:, 0] = labelencoder.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
# Avoiding the Dummy Variable Trap
y = y[:, 1:2]
classifier=XGBClassifier(learning_rate=0.7)
def msg(subject):
    message = re.sub('[^a-zA-Z]', ' ', subject)
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(message)
    corpus1=copy.deepcopy(corpus)
    corpus1.append(message)
    cv = CountVectorizer(max_features=6000)
    X= cv.fit_transform(corpus1).toarray()
    X_train=X[:-1,:]
    X_pred=X[np.size(X,0)-1:np.size(X,0),:]
    y_train=y
    classifier.fit(X_train, np.ravel(y_train,order='C'))
    return classifier.predict(X_pred)
# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Fitting Naive Bayes to the Training set
"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
classifier=XGBClassifier(learning_rate=0.7)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
  """          
# Create a socket object
host=socket.gethostbyname(socket.gethostname());
# Define the port on which you want to connect
port = 7000  
  
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host,port))
s.listen(1) #how many connections can it receive at one time
while True:
    conn, addr = s.accept() #accept the connection
    print ("Connected by: " , addr) #print the address of the person connected
    state_res = conn.recv(1024) #how many bytes of data will the server receive
    message = conn.recv(1024)
    print ("Received: ", str(message,'utf-8'))
    state_res=str(state_res,'utf-8')
    message=str(message,'utf-8')
    print(state_res)
    print(message)
    
    if(state_res[0]=='0'):
        if(state_res[1]=='0'):
            dataset=dataset.append(pd.DataFrame(np.array([['0', message]]), columns=['v1', 'v2']),ignore_index=True)
        else:
            dataset=dataset.append(pd.DataFrame(np.array([['1', message]]), columns=['v1', 'v2']),ignore_index=True)
        dataset.to_csv('spam_messages.csv')
    else:
        temp=msg(message)
        res=int(temp[0])
        if(res==0):
            result="0"
        elif(res==1):
            result="1"
        encoded = result.encode(encoding='utf-8')
        conn.sendall(struct.pack('>i', len(encoded)))
        conn.sendall(encoded)
    print("#1")
    #temp=msg(message)
    print("#2")
    #res=int(temp[0])"""
    """if(res==0):
        result="not spam"
    elif(res==1):
        result="spam"
    print("#3")
    print("#4")
    print("#5")
    print("#6")"""
    conn.close()
# Making the Confusion Matrix
"""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)"""