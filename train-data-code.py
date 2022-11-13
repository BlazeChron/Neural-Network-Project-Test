file1 = open("train-data.txt", "w")
import random

def f(x):
    return x/2 + 0.2


for j in range(100):
    for i in range(5000):
        #file1.write("0,0.5\n")
        x = random.random()
        y = f(x)
        file1.write(str(x) + "," + str(y) + "\n")

file1 = open("test-data.txt", "w")

for j in range(100):
    x = random.random()
    y = f(x)
    file1.write(str(x) + "," + str(y) + "\n")

file1.close()