import numpy as np
#we can also chose to make an assumption about the covariance matrix

class bayes:
    def fit(self,x,y):#Fitting training data and training labels\
        n_samples, n_features = x.shape
        self._classes = np.unique(y)# locating the classes of y
        #initializing the variancees and the mean values
        n_classes = len(self._classes)
        self._mean = np.zeros((n_samples,n_features),dtype=np.float64) #each feature has a mean and a variance
        self._var = np.zeros((n_samples,n_features),dtype = np.float64)
        self._priors = np.zeros(n_classes,dtype=np.float64)
        self.variance_matrices = []
        self.mean = []
        values =[]
        for c in self._classes:
            x_c = x[c==y] #filtering of rows in y of class c
            values.append(x_c)
            self._mean [c,:] = x_c.mean(axis=0) 
            self._var[c,:] = x_c.var(axis=0)
            self._priors[c]= x_c.shape[0]/float(n_samples) #prior probability of y is just the frequency
        #print(values)
        for k in values:
            self.variance_matrices.append(np.cov(k,rowvar = False))
            self.mean.append(k.mean(axis=0))
        self.variance_assumption = 0.5 *np.add(self.variance_matrices[0],self.variance_matrices[1])
        #print(self.variance_matrices)
        #print('\n\n')
    
            
            
        
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
            class_conditional = self.multivariate(x,self.mean[idx],self.variance_assumption)
            posterior = prior + class_conditional
            posteriors.append(posterior)
        #print(posteriors)       
        return self._classes[np.argmax(posteriors)] 
    def multivariate(self,x,mean,variance):
        n = len(x)
        dinominator = np.sqrt(pow(2*np.pi, n) * np.linalg.det(variance))
        mahalanobis = np.dot(np.dot(np.matrix.transpose(np.subtract(x,mean)),np.linalg.inv(variance)),np.subtract(x,mean))
        numerator = np.exp(-0.5 * mahalanobis)
        return np.log(numerator/dinominator)
    

            

import pandas as pd
dataset = pd.read_csv('pima-indians-diabetes.data',header=None)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,8].values 



classifier1 = bayes()
def cross_validation_LOO(xs,ys , classifier):
    correct = 0
    for i in range(len(ys)):
        xt = np.delete(xs,i,0)
        yt = np.delete(ys,i,0)        
        classifier.fit(xt,yt)
        ypred = classifier.predict([xs[i]])
        if ypred == ys[i]:
            correct += 1
    return (correct/len(ys))*100

print('The accuracy of bayes: {}% \n'.format(cross_validation_LOO(X,y,classifier1)))

