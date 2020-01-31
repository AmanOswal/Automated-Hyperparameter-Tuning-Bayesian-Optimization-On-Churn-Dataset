#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for RandomForestClassifier

def hyperopt_train_test(params):
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
space4svm = {
    
    'n_estimators': hp.randint('n_estimators',150),
    #'max_depth' : 1+hp.randint('max_depth', 10),
     'min_samples_split': 1+ hp.randint('min_samples_split' , 3  ),
    'min_samples_leaf' : 1+ hp.randint('min_samples_leaf' , 3  )
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)
'''["n_estimators='warn'", "criterion='gini'", 'max_depth=None', 'min_samples_split=2', 'min_samples_leaf=1', 
    'min_weight_fraction_leaf=0.0', "max_features='auto'", 'max_leaf_nodes=None', 'min_impurity_decrease=0.0', 
    'min_impurity_split=None', 'bootstrap=True',
    'oob_score=False', 'n_jobs=None', 'random_state=None', 'verbose=0', 'warm_start=False', 'class_weight=None'],'''

