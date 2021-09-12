#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np 
import pandas as pd #for Importing Data
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for plotting
from ttictoc import tic,toc
from random import randrange
import random
from math import sqrt
from math import e

dt = pd.read_csv('heart.csv') #Edit accordingly
dt.columns = ['Age', 'Sex', 'Chest_pain_type', 'Resting_blood_pressure', 'Cholesterol', 'Fasting_blood_sugar', 'Rest_ecg', 'Max_heart_rate_achieved',
       'Exercise_induced_angina', 'St_depression', 'St_slope', 'Num_major_vessels', 'Thalassemia', 'Target']
dt.head()#For cheking


# In[3]:


dt.dtypes #As we can see some attributes must be of categorical value. However they are represented as integers.
#We will create dummy variables for categorical variables later. 


# In[3]:


dt.describe() #Checking Properties of the Data


# In[4]:


dt.Target.value_counts() #it shows the number of patients who has heart disease and patients who doesn't have heart disease


# In[5]:



# plot histograms for each variable
dt.hist(figsize = (12, 12))
plt.show()


# In[6]:


#Produce Correlation Matrix to see how each attribution affects each other
plt.figure(figsize=(15,15))
sns.heatmap(dt.corr(),annot=True, linewidths=0.5, fmt='.2f', cmap="coolwarm")
plt.show()
#It can be seen that attributes are not linearly independent of each other. 


# In[17]:


#Now we need to create dummy variables for categorical variables.
#We need to first decide on which attributes are the categorical variables
categorical_val = []
noncategorical_val = []
for column in dt.columns:
    if len(dt[column].unique()) <= 5:  #Educated Guess :D
        categorical_val.append(column)
    else:
        noncategorical_val.append(column)
        
print("Categorical Values:")        
print(categorical_val)
print(f" Total Number : {len(categorical_val)}")
print('<============================================================================================>')
print("Non-Categorical Values:")        
print(noncategorical_val)
print(f" Total Number : {len(noncategorical_val)}")


# In[8]:


#So, there are 9 categorical variables and 5 continuous variables. While continuous variables is already quantified, 
#we have to quantify categorical variables as well. This is done by dummy coding.
#But firstly, lets see how continuous variables are related to target.
import plotly.express as px #For 3d data plot
fig = px.scatter_3d(dt, x='Cholesterol', y='Max_heart_rate_achieved', z='Age', size='St_depression',
              color='Target',opacity=1)
fig.show()
#Couldn't fit in Resting Blood Pressure


# In[9]:


sns.pairplot(dt,hue='Target',diag_kind="hist")
#With this plot, we can see the correlations better!


# In[18]:


categorical_val.remove('Target')
print(categorical_val)


# In[19]:


dataset = pd.get_dummies(dt, columns = categorical_val) #Using Panda's get_dummies command to convert categorical values to dummy variables.
dataset.describe()


# In[20]:


dataset.rename(columns={'Sex_0':'Female'}, inplace=True)
dataset.rename(columns={'Sex_1':'Male'}, inplace=True)
dataset.head()


# In[21]:


#Feature Scaling for better optimization
#Standardization
#We only need to standardize continuous features!

mean = dataset[['Age','Resting_blood_pressure','Cholesterol','Max_heart_rate_achieved','St_depression']].mean(axis=0)
dataset[['Age','Resting_blood_pressure','Cholesterol','Max_heart_rate_achieved','St_depression']] -= mean
std = dataset[['Age','Resting_blood_pressure','Cholesterol','Max_heart_rate_achieved','St_depression']].std(axis=0)
dataset[['Age','Resting_blood_pressure','Cholesterol','Max_heart_rate_achieved','St_depression']] /= std

dataset.head() 


# In[7]:

# Now we will define other functions

def cross_validation(dataset, n_folds):  #Divide the dataset
    dcopy=dataset.copy()
    y = dcopy.Target
    X = dcopy.drop('Target', axis=1)
    dataset_splitX = []
    dataset_splitY = []
    X_copy = X.values.tolist()
    Y_copy = y.values.tolist()
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        foldX = list()
        foldY = list()
        while len(foldX) < fold_size:
            index = randrange(len(X_copy))
            foldX.append(X_copy.pop(index))
            foldY.append(Y_copy.pop(index))
        dataset_splitX.append(foldX)
        dataset_splitY.append(foldY)
    return dataset_splitX,dataset_splitY

def Accuracy(Y_real,Y_predicted): #Accuracy test
    correct = 0
    for i in range(len(Y_real)):
        if Y_real[i] == Y_predicted[i]:
            correct += 1
    return correct / float(len(Y_real)) * 100.0


#For kNN

def Distance_Norm(Vector1, Vector2): #Creating a function to find the norm of difference between two vectors.
    V1=np.array(Vector1)
    V2=np.array(Vector2)
    V=sum((V1-V2)**2)
    Distance=sqrt(V)
    return Distance


def get_Neigh(X_train,Y_train,X_single,num_Neigbour): #Function for getting the closest vectors for a single point and making a list like that
    distances=[]
    for k in range(len(X_train)):
        getDistance=Distance_Norm(X_train[k], X_single)
        distances.append((X_train[k],Y_train[k],getDistance))
    distances=sorted(distances,key=lambda x: x[2])#Sorting the List according to the shortest distance
    neighboursX=[]
    neighboursY=[]
    for i in range(num_Neigbour):
        neighboursX.append(distances[i][0])    
        neighboursY.append(distances[i][1])    
    return neighboursX,neighboursY

def predict_Y(X_train,Y_train, X_single, num_Neigbour):
    neighborsX,neighboursY = get_Neigh(X_train, Y_train, X_single, num_Neigbour)
    counts=[]
    for i in neighboursY:
        counts.append(i)
    ones=counts.count(1)
    zeros=counts.count(0)
    if ones >= zeros:
        return 1
    else:
        return 0
    
def kNN(X_train,Y_train,X_test,num_Neigbour):  #kNN algorithm with the help of other defined functions
    Y_predictions=[]
    for row in X_test:
        predictedY=predict_Y(X_train,Y_train, row, num_Neigbour)
        Y_predictions.append(predictedY)
    return Y_predictions


#Test for kNN algorithm
#In KNN, finding the value of k is not easy. 
#A small value of k means that noise will have a higher influence on the result and a large value make it computationally expensive. 
#k=sqrt(N) where N is the number of feature vectors

'''''
dataset_copy=dataset.copy()
tic()
Scores = algorithm_evaluation(dataset_copy,6,kNN,7)
elapsed = toc()
print('Elapsed time:',elapsed)
print('<===============================================================>')
print('Scores: %s' % Scores)
print('Mean Accuracy: %.3f%%' % (sum(Scores)/float(len(Scores))))

#Elbow method for choosing k
Results=[]
for k in range(30):
    dataset_copy=dataset.copy()
    Scores = algorithm_evaluation(dataset_copy,6,kNN,(k+1))
    Result=(sum(Scores)/float(len(Scores)))
    Results.append(Result)

plt.plot(Results)
print(Results.index(max(Results)))
'''''

# In[8]:


# Support Vector Machine


def gradient_function(W, X, Y, C):
    summ = 0
    for i in range(0, len(X)):
        if (1 - (Y[i] * np.dot(W, X[i]))) > 0:
            summ += (C * Y[i] * X[i])
    
    #  result of gradient of the objective function
    return W - summ


# stochastic gradient descent
def gradient_descent(maxR, gamma, X, Y, C, tol):
    # give weights
    W = np.zeros(len(X[0]))
    for k in range(0, maxR):
        # shuffle the list
        shuffle_list = list(zip(X,Y))
        random.shuffle(shuffle_list)
        X, Y = zip(*shuffle_list)

        grad = gradient_function(W, X, Y, C)
        diff = -grad * gamma
        
        if np.all(np.abs(diff) <= tol):
            return W
       
        else:
            W = W + diff        
    return W

def svm_predict(X,W):
    results = []
    for col in X:
        ans = np.dot(W,col)
        if ans >=0:
            results.append(1)
        else:
            results.append(-1)
    return results


# Number of max repetitions for gradient descent
maxR = 200
# Learning rate of the gradient descent
gamma = 0.0001
# Tolerance to check if the gradient converges
tol = 1e-06


def SVM(X_train, Y_train, X_test, C, maxR, gamma, tol):
    for yi in range(len(Y_train)):
        if Y_train[yi] == 0:
            Y_train[yi] = -1
    X_test = np.array(X_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # Add 1 to every row for bias term
    X_train = np.insert(X_train,0,1,axis = 1)
    X_test = np.insert(X_test,0,1,axis = 1)
    # get the weights of SVM
    W = gradient_descent(maxR, gamma, X_train, Y_train, C, tol)
    # for train accuracy
    train_res = svm_predict(X_train, W)
    # predict the test values
    result = svm_predict(X_test, W)
    return train_res, result



# In[55]:


#For Shallow Neural Network

#Initialize Network
def initialization_of_network(n_inputs, n_hidden, n_outputs):
    total_network=[]
    hidden_layers=[{"Weights": [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] #Extra Weight for bias
    output_layers = [{'Weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)] #Extra Weight for bias
    #So, first component is the weights of the hidden layer
    total_network.append(hidden_layers)
    #So, second component is the weights of the output layer
    total_network.append(output_layers)
    return total_network


#Forward Propagation
def neuron_activation(inputs,neuron_weights):
    activation=neuron_weights[-1]
    for i in range(len(inputs)):
        activation += inputs[i]*neuron_weights[i]
        transfer=1/(1 + e**(-activation))
    return transfer  #This is for each neuron

#Activation function is the sigmoid function!
#output = 1 / (1 + e^(-activation))

'''
def transfer_func(activation_output):
    transfer=1/(1 + e**(-activation_output))
    return transfer
'''
def forward_propagation(network, inputs):
    inputs=inputs
    for layer in network:
        new_input=[]
        for neuron in layer: 
            weights=neuron["Weights"]
            '''
            activation=neuron_activation(inputs,weights)
            neuron["Outputs"]=transfer_func(activation)
            '''
            neuron["Outputs"]=neuron_activation(inputs,weights)
            new_input.append(neuron["Outputs"])
            inputs=new_input
    return inputs #Gives the final output

#Back Propagate Error
def transfer_derivative(output):
    return output * (1.0 - output) #Derivative of sigmoid

def backward_propagation_error(network, expected_output):
    #We will go backwards
    for index in reversed(range(len(network))):
        layer=network[index]
        errors=[]
        if index == len(network)-1:
            for neuron in layer:
                errors.append(expected_output-neuron["Outputs"])
        else:
            for j in range(len(layer)):
                error=0.0
                for neuron in network[index+1]:
                    error += (neuron["Weights"][j]*neuron["Delta"])
                errors.append(error)
        for k in range(len(layer)):
            neuron=layer[k]
            neuron['Delta'] = errors[k] * transfer_derivative(neuron['Outputs'])
            
#Train Network
#We have to update the weights, we do that by weight = weight + learning_rate (new variable) * error(stored in delta) * input
def update_weights(network, input_row , l_rate):
    for index in range(len(network)):
        inputs=input_row
        if not index == 0:
            inputs = [neuron['Outputs'] for neuron in network[index - 1]]
        for neuron in network[index]:
            for k in range(len(inputs)):
                neuron["Weights"][k]=(1-0.003*l_rate)*neuron["Weights"][k]+(l_rate*neuron["Delta"]*inputs[k])
            neuron['Weights'][-1] = (1-0.003*l_rate)*neuron['Weights'][-1]+l_rate * neuron['Delta'] #For bias
        
#       
def Network_Train(network,train_input,l_rate,n_epoch,train_output):
    for epoch in range(n_epoch):
        total_error=0
        for index in range(len(train_input)):
            input_row=train_input[index]
            expected_output=train_output[index]
            output=forward_propagation(network, input_row)
            total_error += (expected_output-output[0])**2
            backward_propagation_error(network, expected_output)
            update_weights(network, input_row, l_rate)
        # error=sqrt(total_error)
        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, error))


def Prediction(network,input_row):
    output=forward_propagation(network, input_row)
    if output[0] >0.5:
        return 1.0
    else:
        return 0.0

#Stochastic Gradient Descent
def back_propagation(trainX, trainY, testX, l_rate, n_epoch, n_hidden):
    n_inputs = len(trainX[0]) 
    n_outputs = 1 
    network = initialization_of_network(n_inputs, n_hidden, n_outputs)
    Network_Train(network, trainX, l_rate, n_epoch, trainY)
    train_predictions = []
    for row in trainX:
        prediction = Prediction(network, row)
        train_predictions.append(prediction)
    test_predictions = []
    for row in testX:
        prediction = Prediction(network, row)
        test_predictions.append(prediction)
    return train_predictions, test_predictions



# In[ ]:

# split test set from dataset
dd = dataset.copy()
train_df = dd.sample(frac = 0.7) 
test_df = dd.drop(train_df.index)
x_val, y_val = cross_validation(train_df.copy(), 6)

train_acc_list1 = []
test_acc_list1 = []
N = np.arange(8,25,2)
for n in N:
    train_acc1 = 0
    test_acc1 = 0
    for ind in range(0, len(x_val)):
        X_test = x_val[ind]
        Y_test = y_val[ind]
        X_train = []
        Y_train = []
        for i in range(len(x_val)):
            if i != ind:
                X_train += x_val[i]
                Y_train += y_val[i]
        trainp, testp = back_propagation(X_train, Y_train, X_test, 0.7, 120, n)
        train_acc1 += Accuracy(Y_train, trainp)
        test_acc1 += Accuracy(Y_test, testp)
    trainacc = train_acc1 / len(x_val)
    testacc = test_acc1 / len(x_val)
    train_acc_list1.append(trainacc)
    test_acc_list1.append(testacc)
plt.plot(N, train_acc_list1)
plt.plot(N, test_acc_list1)
maxtest1 = max(test_acc_list1)
optimal_n = test_acc_list1.index(maxtest1)
print(f"The optimal value of n_hidden is: {N[optimal_n]}")

plt.figure
plt.plot(N, train_acc_list1)
plt.plot(N, test_acc_list1)
plt.xlabel("n_hidden")
plt.ylabel("%acc")
plt.title("Parameter tuning of n_hidden for SNN")
plt.legend(['Train accuracy', 'Test accuracy'])


# find the optimal C, the lagrange multiplier of SVM
# cross validation to find the best value of C between 5 and 150
for fold in y_val:    
    for yi in range(len(fold)):
        if fold[yi] == 0:
            fold[yi] = -1
train_acc_list2 = []
test_acc_list2 = []
R = np.arange(5, 155, 5)
for c in R:
    train_acc2 = 0
    test_acc2 = 0
    for ind in range(0, len(x_val)):
        X_test = x_val[ind]
        Y_test = y_val[ind]
        X_train = []
        Y_train = []
        for i in range(len(x_val)):
            if i != ind:
                X_train += x_val[i]
                Y_train += y_val[i] 
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        # Add 1 to every row for bias term
        X_train = np.insert(X_train,0,1,axis = 1)
        X_test = np.insert(X_test,0,1,axis = 1)

        W = gradient_descent(maxR, gamma, X_train, Y_train, c, tol)
        trainres = svm_predict(X_train, W)
        train_acc2 += Accuracy(Y_train, trainres)
        testres = svm_predict(X_test, W)
        test_acc2 += Accuracy(Y_test, testres)
    trainacc = train_acc2 / len(x_val)
    testacc = test_acc2 / len(x_val)
    train_acc_list2.append(trainacc)
    test_acc_list2.append(testacc)

plt.figure
plt.plot(R, train_acc_list2)
plt.plot(R, test_acc_list2)
plt.xlabel("C")
plt.ylabel("%acc")
plt.title("Parameter tuning of C for SVM")
plt.legend(['Train accuracy', 'Test accuracy'])

maxtest2 = max(test_acc_list2)
optimal_c = test_acc_list2.index(maxtest2)
print(f"The optimal value of C is: {R[optimal_c]}")


# In[ ]:

# one test to rule them all

def final_results(train_df,test_df):
    
    train_df1 = train_df.copy()
    train_setY1 = train_df1.Target.values.tolist()
    train_setX1 = train_df1.drop('Target', axis=1).values.tolist()
    train_df2 = train_df.copy()
    train_setY2 = np.array(train_df2.Target.values.tolist())
    train_setX2 = np.array(train_df2.drop('Target', axis=1).values.tolist())
    train_df3 = train_df.copy()
    train_setY3 = train_df3.Target.values.tolist()
    train_setX3 = train_df3.drop('Target', axis=1).values.tolist()
    
    test_df1 = test_df.copy()
    test_setY1 = test_df1.Target.tolist()
    test_setX1 = test_df1.drop('Target', axis=1).values.tolist()
    test_df2 = test_df.copy()
    test_setY2 = np.array(test_df2.Target.values.tolist())
    test_setX2 = np.array(test_df2.drop('Target', axis=1).values.tolist())
    test_df3 = test_df.copy()
    test_setY3 = test_df3.Target.values.tolist()
    test_setX3 = test_df3.drop('Target', axis=1).values.tolist()
    
    # test of kNN
    tic()
    kNN_predict = kNN(train_setX1, train_setY1, test_setX1, 7)
    kNN_time = toc()
    kNN_accuracy = Accuracy(test_setY1, kNN_predict)
    print("kNN elapsed time:", kNN_time)
    print("kNN accuracy:", kNN_accuracy)
    
    # test of SVM
    tic()
    svm_train_predict, svm_test_predict = SVM(train_setX2, train_setY2, test_setX2, R[optimal_c], maxR, gamma, tol)
    svm_time = toc()
    for yi in range(len(svm_test_predict)):
        if svm_test_predict[yi] == -1:
            svm_test_predict[yi] = 0
    svm_train_accuracy = Accuracy(train_setY2, svm_train_predict)
    svm_test_accuracy = Accuracy(test_setY2, svm_test_predict)
    print("SVM elapsed time:", svm_time)
    print("SVM train accuracy:", svm_train_accuracy)
    print("SVM test accuracy:", svm_test_accuracy)
    
    # test of SNN
    tic()
    snn_train_predict, snn_test_predict = back_propagation(train_setX3, train_setY3, test_setX3, 0.7, 200, N[optimal_n]) #Used heuristics for these numbers
    snn_time = toc()
    for yi in range(len(train_setY3)):
        if train_setY3[yi] == -1:
            train_setY3[yi] = 0
    snn_train_accuracy = Accuracy(train_setY3, snn_train_predict)
    snn_test_accuracy = Accuracy(test_setY3, snn_test_predict)
    print("SNN elapsed time:", snn_time)
    print("SNN train accuracy:", snn_train_accuracy)
    print("SNN test accuracy:", snn_test_accuracy)
    
final_results(train_df,test_df)
    
