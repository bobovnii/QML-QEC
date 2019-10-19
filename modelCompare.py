# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

models = []
results = []
names = []
seed = 8

# columnData = ['Sub Circuit', 'Input State', 'Output State', 'Theta']
# data = [['QFT', [0,1], [1,0], 0.5*math.pi],
# 		 ['QFT', [0.5,0.5], [1,0], 0.5*math.pi],
# 		 ['QFT', [0,1], [0,1], 0],
# 		 ['QFT', [0.5,0.5], [0,1], 0]]

#Mock Data
#X:= list of qubit state input,output pairs [input0,input1,output0,output1], Y:= list of theta parameters for each decoder unitary
X = [[0,1,0,0.99],[1,0,0.99,0.2],[1,0,0.99,0.1],[1,0,0.99,0.05],[1,0,0.99,0.002],[1,0,1,0]]
Y = [0.23,0.43,0.3,0.2,0.1,0]

#Replace with distance measure scoring
def performance_metric(y_true, y_predict):
    score = r2_score(y_true,y_predict)
    return score

scoring = make_scorer(performance_metric)

#Prepare and compare various models. addModels is a dictionary object : {'DT':  DecisionTreeRegressor(), ...}
def set_models(addModels):
	models.append(('DT', DecisionTreeRegressor()))
	models.append(('RD', Ridge()))
	models.append(('LA', Lasso()))
	models.append(('EN', ElasticNet()))

	for model in addModels:
		models.append((model, addModels[model]))

set_models({})

def compare_scores(n_splits_val, models, X, Y):
	for name, model in models:
		kfold = model_selection.KFold(n_splits=n_splits_val, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

compare_scores(3, models, X, Y)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
