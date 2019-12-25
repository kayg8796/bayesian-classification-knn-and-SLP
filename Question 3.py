import numpy as np
class naive_bayes:
    def fit(self,x,y):#Fitting training data and training labels\
        n_samples, n_features = x.shape
        self._classes = np.unique(y) #unique classes of y
        n_classes = len(self._classes)
        self._mean = np.zeros((n_samples,n_features),dtype=np.float64) #each feature has a mean and a variance
        self._var = np.zeros((n_samples,n_features),dtype = np.float64)
        self._priors = np.zeros(n_classes,dtype=np.float64)
        for c in self._classes:
            x_c = x[c==y] #filtering of rows in y of class c
            self._mean [c,:] = x_c.mean(axis=0) 
            self._var[c,:] = x_c.var(axis=0)
            self._priors[c]= x_c.shape[0]/float(n_samples) #prior probability of y is just the frequency
            #print(self._priors[c])
        
    def predict(self,X):#predicting the test samples
        y_pred = [self._predict(x) for x in X]
        #print('next prior set')
        return y_pred
    def _predict(self,x):
        #now we have to calculate the posterior probability(need the class condition and prior prob) and then chose the highest
        posteriors =[]
        
        for idx,c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            #print('{} and {}\n'.format(idx,c))
            class_conditional = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        #print(posteriors)
            
        return self._classes[np.argmax(posteriors)]
            
    def _pdf(self,class_idx,x):
        #applying the gaussian formular
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean)**2) /( 2*var))
        dinominator = np.sqrt(2*np.pi*var)
        #print(numerator/dinominator)
        return numerator/dinominator
        
import pandas as pd
dataset = pd.read_csv('pima-indians-diabetes.data',header=None)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,8].values 

classifier1 = naive_bayes()
def cross_validation_LOO(xs,ys , classifier):
    correct = 0
    for i in range(len(ys)):
        xt = np.delete(xs,i,0)
        yt = np.delete(ys,i,0)        
        classifier.fit(xt,yt)
        ypred = classifier.predict([xs[i]])
        #kc.append(ypred)
        if ypred == ys[i]:
            correct += 1
    return correct/len(ys)

#testing for sklearn
from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()

print('The accuracy of our Naive_bayes classifier is : {}\n'.format(cross_validation_LOO(X,y,classifier1)))
print('The accuracy of our Naive_bayes classifier from scikit learn is : {}\n'.format(cross_validation_LOO(X,y,classifier2)))



