import pandas as pd
import numpy as np
dataset = pd.read_csv('iris.data',header=None)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values 

def labelEncoder(q):
    cl = np.unique(q)
    for idx , w in enumerate(cl):
        q[np.where(q==w)]=idx
    return np.array([int(k) for k in q])

y = labelEncoder(y)
classes = np.unique(y)
class_sets = []
for c in classes:
    x_c = X[c==y]
    class_sets.append(x_c)
class1 = class_sets[0]
class2 = class_sets[1]
class3 = class_sets[2]

def get_weight(class1,class2,class3):    
    t1=1
    t2=-1
    oness =np.ones(len(class1))
    class1 = np.c_[class1,-1 * oness]
    class2n33 = np.concatenate(([class2,class3]))
    class2n3=np.c_[(class2n33,-1 * np.ones(len(class2n33)))]
    class1t = [k*t1 for k in class1]
    class2n3t = [k*t2 for k in class2n3]
    classtrain = np.concatenate((class1t,class2n3t))
    w = [0,0,0,0,0]
    counter = 0
    while counter < 1000:
        k=0
        for i in range(len(classtrain)):
            if np.dot(w,classtrain[i]) > 0:
                k += 1
            else:
                w = np.add(w,classtrain[i])
        if k == len(classtrain):
            break
        counter += 1
    return w , counter

w1,counter1 = get_weight(class1,class2,class3)
w2 , counter2= get_weight(class2,class3,class1)
w3 ,counter3= get_weight(class3,class2,class1)

if counter1 == 1000:
    print('The iris-setosa class is NOT linearly separable from the other two\n')
else:
    print('The iris-setosa class is linearly separable from the other two classes\n')
if counter2 == 1000:
    print('The iris-versicolor class is NOT linearly separable from the other two\n')
else:
    print('The iris-versicolor class is linearly separable from the other two classes\n')
if counter3 == 1000:
    print('The iris-verginia class is NOT linearly separable from the other two\n')
else:
    print('The iris-verginia class is linearly separable from the other two classes\n')
    
print('\n')


    


        
        
