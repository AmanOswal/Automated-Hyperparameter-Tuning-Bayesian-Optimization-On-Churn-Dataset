#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
def hyperopt_train_test(params):
    clf = GaussianNB(**params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
space4GNB = {
    'var_smoothing' :hp.uniform('var_smoothing', 10**(-11),10**(-7))
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4GNB, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)

