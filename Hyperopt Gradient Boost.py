#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for Gradient Boosting

def hyperopt_train_test(params):
    clf = GradientBoostingClassifier(**params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
space4svm = {
    'learning_rate': hp.uniform('C', 0, 0.5),
    'n_estimators': hp.randint('n_estimators',150),
    'subsample': hp.uniform('subsample', 0, 1),
    'max_depth' : 1+  hp.randint('max_depth', 10)
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)
 

