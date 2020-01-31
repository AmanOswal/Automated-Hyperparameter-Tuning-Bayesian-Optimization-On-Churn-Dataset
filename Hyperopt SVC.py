#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for SVC
def hyperopt_train_test(params):
    clf = SVC(**params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
space4svm = {
    'C': hp.uniform('C', 0, 5),
    'kernel': hp.choice('kernel', ['sigmoid', 'rbf', 'linear']),
    'gamma': hp.uniform('gamma', 0, 5)    
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)

