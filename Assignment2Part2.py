# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:31:20 2019

@author: Rahul
"""

import load_data
import numpy as np
import time
#from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score




# return DataSet class
data = load_data.read_data_sets(one_hot=True)
BATCH_SIZE = 84420
# get train data and labels by batch size
train_x, train_label = data.train.next_batch(BATCH_SIZE)
train_labels = []

#------------------------------------------PreProcessing train Labels------------------------------------------
for row in train_label:
	if(row[0] == 1):
		train_labels.append(0)
	elif (row[1] == 1):
		train_labels.append(1)
	else:
		train_labels.append(2)

#---------------------------------------------------------------------------------------------------------------
		
		
# get test data
test_x = data.test.data

# get test labels
test_label = data.test.labels



test_labels = []

#---------------------------------------------PreProcessing test Labels---------------------------------------
for row in test_label:
	if(row[0] == 1):
		test_labels.append(0)
	elif (row[1] == 1):
		test_labels.append(1)
	else:
		test_labels.append(2)
#-------------------------------------------------------------------------------------------------------------

#---------------------------------------Calculation without feature extraction-----------------------------------

clf = DecisionTreeClassifier(random_state=0)
#---------------------------------------Training time without feature extraction-------------------------------- 

start = time.time()
clf.fit(train_x,train_labels)
end = time.time()

print("Training time without feature extraction: ", (end-start))
#---------------------------------------------------------------------------------------------------------------

predictedClass = clf.predict(test_x)

#---------------------------------------Model Accuracy---------------------------------------------------------
print("Model Accuracy : ",accuracy_score(test_labels, predictedClass)*100,"%")

#---------------------------------------------------------------------------------------------------------------
number_of_positive_emotions = 0
number_of_neutral_emotions = 0
number_of_negative_emotions = 0
for j in range(len(test_labels)):
	if(test_labels[j] == 0 ):
		number_of_positive_emotions+=1
	elif(test_labels[j] == 1):
		number_of_neutral_emotions+=1
	elif(test_labels[j] == 2):
		number_of_negative_emotions+=1
		

		
number_of_correct_positive_emotions = 0
number_of_correct_neutral_emotions = 0
number_of_correct_negative_emotions = 0		
for i in range(len(predictedClass)):
	if predictedClass[i] == test_labels[i] and test_labels[i] == 0:
		number_of_correct_positive_emotions+=1
	elif predictedClass[i] == test_labels[i] and test_labels[i] == 1:
		number_of_correct_neutral_emotions+=1
	elif predictedClass[i] == test_labels[i] and test_labels[i] == 2:
		number_of_correct_negative_emotions+=1

#-----------------------------Calculating Top 1 Accuracy-------------------------------------------------------
accuracy = []
accuracy.append(number_of_correct_positive_emotions/number_of_positive_emotions)
accuracy.append(number_of_correct_neutral_emotions/number_of_neutral_emotions)
accuracy.append(number_of_correct_negative_emotions/number_of_negative_emotions)
print("Top-1 Accuracy without feature extraction: ",max(accuracy)*100, "%" )

#--------------------------------------------------------------------------------------------------------------       


#----------------------------------------Feature Extraction LDA-----------------------------------------------
        
dataset = np.concatenate((train_x,test_x), axis=0)
labelDataset = np.concatenate((train_labels,test_labels), axis=0)
transformer = LinearDiscriminantAnalysis(n_components=2)
X_transformed_dataset = transformer.fit(dataset,labelDataset).transform(dataset)

train_x = X_transformed_dataset[0:len(train_x)]
test_x = X_transformed_dataset[ len(train_x): ( len(train_x)+len(test_x) ) ]

#-----------------------------------------------------------------------------------------------------------



#---------------------------------Calculation with feature extraction--------------------------------------- 
clf = DecisionTreeClassifier(random_state=0)

#---------------------------------------Training time with feature extraction-------------------------------- 

start = time.time()
clf.fit(train_x,train_labels)
end = time.time()

print("Training time with feature extraction: ", (end-start))
#---------------------------------------------------------------------------------------------------------------
predictedClass = clf.predict(test_x)
#---------------------------------------------Model Accuracy with feature extraction--------------------------------------------------------
#print("Model Accuracy with feature extraction: ",accuracy_score(test_labels, predictedClass)*100 )
#--------------------------------------------------------------------------------------------------------------
number_of_positive_emotions = 0
number_of_neutral_emotions = 0
number_of_negative_emotions = 0
for j in range(len(test_labels)):
	if(test_labels[j] == 0 ):
		number_of_positive_emotions+=1
	elif(test_labels[j] == 1):
		number_of_neutral_emotions+=1
	elif(test_labels[j] == 2):
		number_of_negative_emotions+=1
		

		
number_of_correct_positive_emotions = 0
number_of_correct_neutral_emotions = 0
number_of_correct_negative_emotions = 0		
for i in range(len(predictedClass)):
	if predictedClass[i] == test_labels[i] and test_labels[i] == 0:
		number_of_correct_positive_emotions+=1
	elif predictedClass[i] == test_labels[i] and test_labels[i] == 1:
		number_of_correct_neutral_emotions+=1
	elif predictedClass[i] == test_labels[i] and test_labels[i] == 2:
		number_of_correct_negative_emotions+=1

#-----------------------------Calculating Top 1 Accuracy-------------------------------------------------------
accuracy = []
accuracy.append(number_of_correct_positive_emotions/number_of_positive_emotions)
accuracy.append(number_of_correct_neutral_emotions/number_of_neutral_emotions)
accuracy.append(number_of_correct_negative_emotions/number_of_negative_emotions)
print("Top-1 Accuracy with feature extraction: ",max(accuracy)*100, "%" )

#---------------------------------------------------------------------------------------------------------------



