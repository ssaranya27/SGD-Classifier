# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Dataset: Load the Iris dataset from the sklearn.datasets library.

2.Prepare Data: Create a pandas DataFrame from the dataset, assigning feature columns and a target column.

3.Split Data: Split the DataFrame into features (X) and target labels (y), then divide the data into training and testing sets.

4.Train Model: Initialize the SGDClassifier with default parameters and train it on the training set.

5.Evaluate Model: Make predictions on the test set, calculate accuracy, and generate the confusion matrix to evaluate model performance.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SARANYA S.
RegisterNumber:  212223220101
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
#load the iris dataset
iris = load_iris()

#create a pandas dataframe
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target'] = iris.target

#display the first few rows of the dataset
print(df.head())
#split the data into features (x) and target (y)
x=df.iloc[:,:-1]
print(x)
y=df.iloc[:-1]
y
x=df.drop('target',axis=1)
y = df['target']
#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/2,random_state=42)
#create an sgd classifier with default parameters
sgd_clf =SGDClassifier(max_iter=1000,tol=1e-3)
#spliting the data training and testing
sgd_clf.fit(x_train,y_train)
y_pred = sgd_clf.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
#calculate confusion matrix
cm = confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(cm)
```
## Output:

![image](https://github.com/user-attachments/assets/486b9642-cbf1-4952-bb33-14c7d0feecbd)

ACCURACY AND CONFUSION MATRIX

![image](https://github.com/user-attachments/assets/9651090e-6a7d-4733-9836-228e91188686)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
