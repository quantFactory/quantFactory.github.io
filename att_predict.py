# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:42:39 2018

@author: ali
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:49:42 2018

@author: ali
"""

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd  
import os 




path_ = os.getcwd()
print(path_)


seed = 10
_input_dim = 35
kernel_init = "normal"
activation_fun = 'tanh'
_loss = 'binary_crossentropy'
_optimizer = 'adam'
_metrics  = 'accuracy' 


np.random.seed(seed)
#reading data
training_X = pd.read_csv("./data/attr.csv")
training_X.shape
training_X_ds = training_X.values
list(training_X.columns)


# training and test set detemination
X =  training_X_ds[:,0:35].astype(float)
Y =  training_X_ds[:,34]




'''
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y)
'''

global model
##baseline
##training deep feed forward 
def create_baseline():
     model= Sequential()
     model.add(Dense(36, input_dim = _input_dim, kernel_initializer= kernel_init,activation = 'relu'))     
     model.add(Dense(16,activation='sigmoid'))     
     model.add(Dense(1, kernel_initializer= "normal", activation='sigmoid'))
     model.compile(loss = _loss,optimizer = 'adam' ,metrics=['accuracy'])
    # model.fit(X,Y,validation_data=(X,Y),validation_split=0.75,shuffle= True,verbose=2,batch_size=100,epochs=100)
     #model.fit(X,Y)
     model.save("./models/pre_trained.h5")
     return model

estimators = []    
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn = create_baseline,shuffle= True,verbose=1,batch_size=100,epochs=100)))
pipline = Pipeline(estimators)        
#estimator = KerasClassifier(build_fn =create_baseline)    
kfold = StratifiedKFold(n_splits =10, shuffle=True, random_state=seed)
results = cross_val_score(pipline, X, Y, cv= kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#model = create_baseline()
model.save("./models/pre_trained.h5")
print("model saved to disk")

    
