import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_curve,roc_curve, average_precision_score, roc_auc_score, auc
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class singleClassifier:
    def __init__(self,clf,XTrain,yTrain,XTest, yTest,cvNum=5):
        self.clf = clf
        self.X = XTrain
        self.y = yTrain
        self.XTest = XTest
        self.yTest = yTest
        self.yPred = np.array([])
        self.yProb = np.array([])
        self.cvNum = cvNum
        self.optimalParams = collections.defaultdict()
        self.optClf  = clf
        self.imbalanceDict  = collections.Counter(self.y)
        self.accuracy = 0.5
        self.confusionMatrix = np.array([])
        self.precision = np.array([])
        self.recall = np.array([])
        self.threshold = np.array([])
        self.featureImportances = pd.DataFrame()

    def sample(self):
        sampler = RandomUnderSampler()
        self.X,self.y= sampler.fit_resample(self.X,self.y)


    def search(self,params,method='grid'):
        if method == 'grid':
            cvSearch  = GridSearchCV(self.clf,params)
        if method == 'random':
            cvSearch  = RandomizedSearchCV(self.clf,params)
        cvSearch.fit(self.X, self.y)
        self.optimalParams = cvSearch.best_params_
        self.optClf = cvSearch.best_estimator_
        print([cvSearch.best_params_, cvSearch.best_score_])

    def fit(self):
        self.optClf.fit(self.X,self.y)

    def predict(self):
        self.yPred = self.optClf.predict(self.XTest)
        self.yProb = self.optClf.predict_proba(self.XTest)
        self.yProb = self.yProb[:,1]

    def evaluate(self):
        self.accuracy = accuracy_score(self.yTest,self.yPred)
        self.confusionMatrix = confusion_matrix(self.yTest,self.yPred)
        print(self.accuracy)
        print(self.confusionMatrix)

    def plot(self):
        self.precision,self.recall,self.threshold = precision_recall_curve(self.yTest,self.yProb)
        pr,rc,thr = self.precision,self.recall,self.threshold
        pr,rc = pr[:-1],rc[:-1]

        fig,ax = plt.subplots()
        precision, = ax.plot(thr,pr,color = 'steelblue',label = 'Precision')
        ax.set_xlim([0.0,1.0])
        ax.set_xlabel ('Threshold')
        ax.set_ylabel('Precision')
        ax.set_title(f'Area Under PR/RC Curve: {auc(thr,np.minimum(pr,rc))}')
        ax2 = ax.twinx()
        recall, = ax2.plot(thr,rc,color='darkorange',label = 'Recall')
        ax2.set_ylabel('Recall')
        plt.legend(handles = [precision,recall])
        plt.show()
        plt.close()

        fpr,tpr,thr = roc_curve(self.yTest,self.yProb)

        plt.figure()
        lw = 2
        plt.plot(fpr,tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % roc_auc_score(self.yTest,self.yProb))
        plt.plot([0, 1], [0, 1], color="steelblue", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()
        plt.close()

        plt.figure()
        lw = 2
        plt.plot(rc, pr, color="steelblue", lw=lw)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Average Precision %0.4f" % average_precision_score(self.yTest,self.yProb))
        plt.show()
        plt.close()

    def importances(self):
        importances = self.optClf.feature_importances_
        self.featureImportances['Features'] = self.X.columns
        self.featureImportances['Importances'] = minmax_scale(importances,(0,100))
        self.featureImportances.sort_values(by='Importances',ascending=False,inplace=True)
        return self.featureImportances