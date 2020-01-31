#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for XGBoost
#uSE thIS
def hyperopt_train_test(params):
    clf = xgb.XGBClassifier(**params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
space4xgb = {
    'max_depth': 5+hp.randint('max_depth', 5),
    'gamma' : hp.uniform('gamma',0,1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.3,1),
    'subsample' : hp.uniform('subsample', 0.5,1),
    
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4xgb, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)

