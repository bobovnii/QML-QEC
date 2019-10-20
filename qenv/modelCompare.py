"""
Compare Algorithms
"""

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
from sklearn.metrics import make_scorer


class Compare:
    def __init__(self, seed=8):
        self.models = list()
        self.results = list()
        self.names = list()
        self.seed = seed
        self.scoring = make_scorer(self.performance_metric)

    def _reset(self):
        self.models = list()
        self.results = list()
        self.names = list()
        self.scoring = make_scorer(self.performance_metric)

    def performance_metric(self, y_true, y_predict):
        # Replace with distance measure scoring
        score = np.linalg.norm(y_true - y_predict)
        return score

    def run(self, X=None, Y=None, n_splits_val=3, plot=False):
        """
        Mock Data
        X:= list of qubit state input,output pairs [input0,input1,output0,output1],
        Y:= list of theta parameters for each decoder unitary

        columnData = ['Sub Circuit', 'Input State', 'Output State', 'Theta']
        data = [['QFT', [0,1], [1,0], 0.5*math.pi],
                ['QFT', [0.5,0.5], [1,0], 0.5*math.pi],
                ['QFT', [0,1], [0,1], 0],
                ['QFT', [0.5,0.5], [0,1], 0]]
        """
        self._reset()

        if X is None:
            X = [
                [0, 1, 0, 0.99],
                [1, 0, 0.99, 0.2],
                [1, 0, 0.99, 0.1],
                [1, 0, 0.99, 0.05],
                [1, 0, 0.99, 0.002],
                [1, 0, 1, 0],
            ]

        if Y is None:
            Y = [0.23, 0.43, 0.3]

        self.set_models({})

        self.compare_scores(n_splits_val, self.models, X, Y)

        if plot:
            # boxplot algorithm comparison
            fig = plt.figure()
            fig.suptitle("Algorithm Comparison")
            ax = fig.add_subplot(111)
            plt.boxplot(self.results)
            ax.set_xticklabels(self.names)
            plt.show()

    def set_models(self, addModels):
        # Prepare and compare various models. addModels is a dictionary object : {'DT':  DecisionTreeRegressor(), ...}
        self.models.append(("DT", DecisionTreeRegressor()))
        self.models.append(("RD", Ridge()))
        self.models.append(("LA", Lasso()))
        self.models.append(("EN", ElasticNet()))

        for model in addModels:
            self.models.append((model, addModels[model]))

    def compare_scores(self, n_splits_val, models, X, Y):
        for name, model in models:
            kfold = model_selection.KFold(n_splits=n_splits_val, random_state=self.seed)
            cv_results = model_selection.cross_val_score(
                model, X, Y, cv=kfold, scoring=self.scoring
            )
            self.results.append(cv_results)
            self.names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
