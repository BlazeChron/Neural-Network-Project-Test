#doesn't have to be perfect, it's a first trial run. You have to program it again later anyway

import numpy as np
import csv


def sigmoid(Z): #will return elementwise
    return 1 / (1 + np.exp(-Z))

def sigmoidPrimeStar(A):    #note takes in A not Z
    oNE = np.divide(A, A) #creates a matrix of 1s same size as A
    oNE = oNE - A
    return np.multiply(A, oNE)

#load training data
trainFile = open("train-data.txt", "r")
reader = csv.reader(trainFile, delimiter=",")
x = list(reader)
trainData = np.array(x).astype(float)
trainFile.close()

#initialise variables

weightFile = open("starting-weights_1.txt", "r")
reader = csv.reader(weightFile, delimiter=",")
w = list(reader)
W_1 = np.array(w).astype(float)
W_1 = np.transpose(W_1)
weightFile = open("starting-weights_2.txt", "r")
reader = csv.reader(weightFile, delimiter=",")
w = list(reader)
W_2 = np.array(w).astype(float)
W_2 = np.transpose(W_2)
weightFile.close()

print(W_1)
print(W_2)

#weight oscillation mitigation
dW_1old = 0
dW_2old = 0

count = 0
length = len(trainData)

for j in range(2):  #number of times train dataset
    for i in trainData:

        #forward prop
        X = i
        a_0 = [[sigmoid(X[0])]]
        a_0a = [np.append(a_0[0], 1)]
        z_1 = np.dot(a_0a, W_1)
        a_1 = sigmoid(z_1)
        a_1a = [np.append(a_1[0], 1)]
        z_2 = np.dot(a_1a, W_2)
        a_2 = sigmoid(z_2)

        #backward prop
        T = [[sigmoid(X[1])]]
        dz_2 = np.multiply((T-a_2), sigmoidPrimeStar(a_2))
        dW_2 = np.dot(np.transpose(a_1a), dz_2)               
        dz_1 = np.multiply((np.dot(dz_2, np.transpose(W_2[:len(W_2)-1]))), sigmoidPrimeStar(a_1)) #W_2[:len(W_2)-1] removes the bias weights as they are not relevant to previous layer
        dW_1 = np.dot(np.transpose(a_0a), dz_1)

        #update weights
        W_1 = W_1 + 0.1 * dW_1 + 0.1 * dW_1old
        W_2 = W_2 + 0.1 * dW_2 + 0.1 * dW_2old
        dW_1old = dW_1
        dW_2old = dW_2

        count += 1
        if (count % 10000 == 0):
            print(str(count/length * 100)+"%")
            print(T-a_2)



#test area

testFile = open("test-data.txt", "r")
reader = csv.reader(testFile, delimiter=",")
x = list(reader)
testData = np.array(x).astype(float)
testFile.close()

for i in testData:
    a_0 = [[sigmoid(i[0])]]
    a_0a = [np.append(a_0[0], 1)]
    z_1 = np.dot(a_0a, W_1)
    a_1 = sigmoid(z_1)
    a_1a = [np.append(a_1[0], 1)]
    z_2 = np.dot(a_1a, W_2)
    a_2 = sigmoid(z_2)
    error = [[sigmoid(i[1])]] - a_2
    print("input: " + str(i) + " output: " + str(z_2) + " expected: " + str(i[1]) + " error: " + str(error) + "\n\n")

print(W_1)
print(W_2)