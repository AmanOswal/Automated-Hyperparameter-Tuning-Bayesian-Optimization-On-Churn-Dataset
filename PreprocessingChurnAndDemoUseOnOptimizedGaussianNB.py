#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the dataset
dataset = pd.read_csv('Churn.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Over_Sampling
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

best= {'C': 2.7029833026257085, 'gamma': 1.090192166611637, 'kernel': 'rbf'}
model = SVC(**best)
a_1=cross_val_score(model, X_train, y_train,cv=10)
a_1.mean()
print("Optimized result from AHT", a_1.mean()*100)
print("Std Dev result from AHT", a_1.std())

# Predicting the Test set results
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc= (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0]+ cm[1][1])
print("Accuracy % on Test Data = " , 100*acc)

# Precision is true positives/total predicted positives
Precision= cm[1][1]/(cm[1][1]+cm[0][1])
print("Precision on Test Data = " ,Precision)
# Precision is true positives/total Actual positives
Recall= cm[1][1]/(cm[1][1]+cm[1][0])
print("Recall on Test Data = " ,Recall)
#F1 score or Fvalue is 2*{Precision*Recall/(Precision + Recall)}
F_value = 2*Precision*Recall/(Precision + Recall)
print("F_value on Test Data = " ,F_value)

