#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for Neural Networks
from keras.wrappers.scikit_learn import KerasClassifier

def model1(n1=1,n2=1,n3=1,a1 = "relu", a2 = "relu", a3 = "sigmoid", losses = "binary_crossentropy", optimizers = "sgd"):
    model = Sequential()
    model.add(Dense(n1, activation = a1))
    if(n2!=0):
        model.add(Dense(n2, activation = a2))
    if(n3!=0):
        model.add(Dense(n3, activation = a3))
    model.add(Dense(2,activation="sigmoid"))
    model.compile(loss = losses, optimizer= optimizers, metrics = ["accuracy"])
    return model
mo = KerasClassifier(build_fn=model1)
def hyperopt_train_test(params):
    clf = KerasClassifier(build_fn=model1 , **params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
space4NN = {
    'n1': 1 + hp.randint('n1',7),
    'n2':  hp.randint('n2',7),
    'n3':  hp.randint('n3',7),
    'a1' : hp.choice('a1',["relu", "sigmoid"]),
    'a2': hp.choice( 'a2',["relu", "sigmoid"]),
    'a3':  hp.choice('a3',["relu", "sigmoid"]),
    'losses': hp.choice('losses',["mse", 'binary_crossentropy']),
    'optimizers': hp.choice('optimizers',["sgd", "adam"])
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4NN, algo=tpe.suggest, max_evals=1, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)        
    

