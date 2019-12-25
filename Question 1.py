import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
class knn():
    def fit(self,X,y,nn):
        self.X = X
        self.y=y
        self.neighbours=nn
    def euclidean_distance(self,x,y):#x1 and x2 are points from column vectors
        sum=0
        for i in range(0,len(x)):
            sum += (x[i] - y[i])**2
        return math.sqrt(sum)    
    def get_neighbours(self,x):# x is the new set of features , X is the feature set, in this case each row has 4 features
        n = self.X.shape[0] #number of rows
        #print(n)
        u_distance = []
        for i in range(0,n):
            u_distance.append(self.euclidean_distance(x,self.X[i]))
        return u_distance  
    def ppredict(self,x):
        uc_distances = self.get_neighbours(x)
        idx = sorted(range(0,len(uc_distances)),key = lambda k: uc_distances[k])
        classes = []
        counter = 0
        for i in idx:
            #print(i)
            classes.append(self.y[i])
            counter += 1
            if counter == self.neighbours :
                break
        categories = Counter(classes)
        descending_table = sorted(categories,key = lambda d:categories[d],reverse = True) # sorts from the highest occuring to the lowest
        #print(descending_table)
        return descending_table[0]       
    def predict(self,xtest):
        ypred = []
        for x in xtest:
            ypred.append(self.ppredict(x))
        return ypred

import pandas as pd
dataset = pd.read_csv('iris.data',header=None)

def labelEncoder(q):
    cl = np.unique(q)
    for idx , w in enumerate(cl):
        q[np.where(q==w)]=idx
    return np.array([int(k) for k in q])

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values #could not view the object on my editor so i turned it into a dataframe
#y = pd.DataFrame(y1)

y = labelEncoder(y)


dataset2 = pd.read_csv('pima-indians-diabetes.data',header=None)
X2 = dataset2.iloc[:,:-1].values
y2 = dataset2.iloc[:,8].values 

classifier1 = knn()
def cross_validation_LOO(xs,ys,classifier,n):
    correct = 0
    for i in range(len(ys)):
        xt = np.delete(xs,i,0)
        yt = np.delete(ys,i,0)        
        classifier.fit(xt,yt,n)
        ypred = classifier.predict([xs[i]])
        #kc.append(ypred)
        if ypred == ys[i]:
            correct += 1
    return correct/len(ys)

  
  
#print('the accuracy of this algorithm is : {}%\n'.format(cross_validation_LOO(X,y,classifier1,15)))
#print('the accuracy of this algorithm is : {}%\n'.format(cross_validation_LOO(X,y,classifier2)))
accuracy=[]
n_neighbours = []
for i in range(len(y)):
    if i%3 == 0:
        continue
    else:
        n_neighbours.append(i)
        accuracy.append(cross_validation_LOO(X,y,classifier1,i))    


accuracy2=[]
n_neighbours2 = np.arange(1,len(y2),50)
for i in n_neighbours2: #this loop is computationally expensive , you can can the lenghth of the n_neighbour list to get a smaller compute time
    accuracy2.append(cross_validation_LOO(X2,y2,classifier1,i))    


plt.plot(n_neighbours,[100*k for k in accuracy])
plt.ylabel('% accuracy')
plt.xlabel('nearest neighbours')
plt.title('KNN implementation for Iris dataset')
plt.show()
print('The maximun accuracy is : {}% \n at k = {}'.format(100*accuracy[np.argmax(accuracy)],n_neighbours[np.argmax(accuracy)]))


plt.plot(n_neighbours2,[100*k for k in accuracy2])
plt.ylabel('% accuracy')
plt.xlabel('nearest neighbours')
plt.title('KNN implementation for pima dataset')
plt.show()
print('The maximun accuracy is : {}% \n at k = {}'.format(100*accuracy2[np.argmax(accuracy2)],n_neighbours2[np.argmax(accuracy2)]))

