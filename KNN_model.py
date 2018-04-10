import pandas as pd

dataset = pd.read_csv('kidney_disease.csv')
    
# X denotes the dataset
X = dataset[['age','bp','sg','al','su']]
# y denotes the classes or labels
y = dataset[['classification']]

#DESCRIBE THE DATASET TO VIEW THE MISSING VALUES
#step 1: view the total number of rows in the dataset
print (len(X))
#step 2: view the missing values in the different columns
print (X.describe())

#Using imputer to fill the missing values
#step 1: call the imputer function from skleran library
from sklearn.preprocessing import Imputer
#step 2: configure the imputer model
model = Imputer(missing_values='NaN', strategy='mean', axis=0)
#step 3: apply the imputer model
X = model.fit_transform(X)

#Convert numpy matrix to pandas dataframe
X = pd.DataFrame(X)

#Now again check the number of missing values in the dataset
print (X.describe())

#Divide the dataset in training and testing data
#step 1: call the train test split method
from sklearn.model_selection import train_test_split
#step 2: split the data using the train test split method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Configure and train the KNN classifier
#step 1:call the knn model from the sklearn library
from sklearn.neighbors import KNeighborsClassifier
#step 2: configure the classifier with number of neighbors required 
#to take the classification decision
clf = KNeighborsClassifier(n_neighbors=3)
#step 3: Train the classifer
clf.fit(X_train,y_train)
#step 4: test the classifier
y_preds = clf.predict(X_test)
#step 5: call the accuracy_score method
from sklearn.metrics import accuracy_score
#step 6: analyze the classification accuracy
print ('Accuracy of the propose model: ')
print (accuracy_score(y_preds,y_test))