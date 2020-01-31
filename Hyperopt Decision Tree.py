#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bayesian Optimization of Hyperparameters for Decision Tree

def hyperopt_train_test(params):
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X_train, y_train, cv=10).mean()
space4DT = {
    'min_weight_fraction_leaf' : hp.uniform('min_weight_fraction_leaf', 0, 0.5),
    'min_samples_split' : 2+ hp.randint('min_samples_split',5),
    'min_samples_leaf' : 1 +  hp.randint('min_samples_leaf', 10)
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4DT, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)
print ('best:')
print (best)
print(trials)
'''"criterion='gini'", "splitter='best'", 'max_depth=None', 'min_samples_split=2', 
 'min_samples_leaf=1', 'min_weight_fraction_leaf=0.0', 'max_features=None', 'random_state=None',
 'max_leaf_nodes=None', 'min_impurity_decrease=0.0', 'min_impurity_split=None', 'class_weight=None', 'presort=False'''

