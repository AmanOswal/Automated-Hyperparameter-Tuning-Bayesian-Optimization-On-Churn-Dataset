#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for KNN
from sklearn.neighbors import KNeighborsClassifier
def hyperopt_train_test(params):
    clf =  KNeighborsClassifier(**params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
spaceKNN = {
    'n_neighbors': 1 + hp.randint('n_neighbours',10),
    "weights" : hp.choice('wts',['uniform','distance']), 
    "algorithm":hp.choice('algo', ['auto', 'ball_tree', 'kd_tree', 'brute']),
    'leaf_size': 20 + hp.randint('ls',20), 
    'p':1+ hp.randint('p', 5)
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, spaceKNN, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)

