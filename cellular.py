#data analysis libraries



import numpy as np
import pandas as pd


pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 37)
pd.set_option('precision', 2)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sbn

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#STEP-2) Read in and Explore the Data
#*********************************************
#It's time to read in our training and testing data using pd.read_csv, and take a first look at the training data using the describe() function.

#import train and test CSV files
train = pd.read_csv('Micro.csv')   #12 columns


#take a look at the training data

print( train.describe() )

print( "\n"  )

print( train.describe(include="all")  )

print(  "\n"  )

print(  "\n\n" , train.columns  )




print()
print( pd.isnull(train).sum()  )

#sbn.barplot(x="last_rech_amt_ma", y="label", data=train)
#plt.show()

for x in range(len(train['aon'])):
    if 'UA' in train['aon'][x]:
        train['aon'][x] = -1

train['aon'] = train['aon'].astype(float)

for x in range(len(train['daily_decr30'])):
    if 'UA' in train['daily_decr30'][x]:
        train['daily_decr30'][x] = -1

train['daily_decr30'] = train['daily_decr30'].astype(float)

for x in range(len(train['daily_decr90'])):
    if 'UA' in train['daily_decr90'][x]:
        train['daily_decr90'][x] = -1

train['daily_decr90'] = train['daily_decr90'].astype(float)





train = train.drop('msisdn',axis = 1)
train = train.drop('pcircle',axis = 1)
train = train.drop('pdate',axis = 1)
train = train.drop('rental30',axis = 1)
train = train.drop('rental90',axis = 1)

from sklearn.model_selection import train_test_split

output_data = train['label']

x_train, x_val, y_train, y_val=train_test_split(
    train, output_data, test_size = 0.20, random_state = 7)

from sklearn.metrics import accuracy_score

#MODEL-1) LogisticRegression
#------------------------------------------
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg  )


#MODEL-2) Gaussian Naive Bayes
#------------------------------------------
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-2: Accuracy of GaussianNB : ", acc_gaussian  )














#MODEL-5) Perceptron
#------------------------------------------
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-5: Accuracy of Perceptron : ",acc_perceptron  )






#MODEL-6) Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree  )







#MODEL-7) Random Forest

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest  )








#MODEL-8) KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn  )








#MODEL-9) Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd )






#MODEL-10) Gradient Boosting Classifier
#------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk )










#Let's compare the accuracies of each model!

models = pd.DataFrame({
    'Model': ['Logistic Regression','Gaussian Naive Bayes',
               'Perceptron',  'Decision Tree',
              'Random Forest', 'KNN','Stochastic Gradient Descent',
              'Gradient Boosting Classifier'],
    'Score': [acc_logreg, acc_gaussian, acc_perceptron,  acc_decisiontree,acc_randomforest,  acc_knn,  acc_sgd, acc_gbk]})


print()
print( models.sort_values(by='Score', ascending=False) )




