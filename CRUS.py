import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_curve,roc_curve, average_precision_score, roc_auc_score, auc
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


class CRUS:
    def __init__(self,clf,XTrain,yTrain,XTest,yTest,features,cvNum=5):
        self.clf = clf
        self.clfs = []
        self.X = XTrain
        self.y = yTrain
        self.rusX = np.array([])
        self.rusy = np.array([])
        self.XTest = XTest
        self.yTest = yTest
        self.yPred = np.array([])
        self.yProb = np.array([])
        self.dfProb = pd.DataFrame()
        self.dfOptimize = pd.DataFrame()
        self.probMatrix = np.array([])
        self.dfImportances = pd.DataFrame()
        self.features = features
        self.featureImportances = pd.DataFrame()
        self.weights = []
        self.cvNum = cvNum
        self.dataLoader = []
        self.optimalParams = collections.defaultdict()
        self.imbalanceDict  = collections.Counter(self.y)
        self.accuracy = 0.5
        self.confusionMatrix = np.array([])
        self.precision = np.array([])
        self.recall = np.array([])
        self.threshold = np.array([])

    def search(self, params, method='grid'):
        if method == 'grid':
            cvSearch = GridSearchCV(self.clf, params, verbose=2,cv=self.cvNum)
        if method == 'random':
            cvSearch = RandomizedSearchCV(self.clf, params, verbose=2,cv=self.cvNum)
        sampler = RandomUnderSampler()
        self.rusX,self.rusy = sampler.fit_resample(self.X, self.y)
        cvSearch.fit(self.rusX,self.rusy)
        self.optimalParams = cvSearch.best_params_
        print([cvSearch.best_params_, cvSearch.best_score_])

    def sample(self):
        self.X, self.y = shuffle(self.X, self.y)
        classes = [(key,value) for key, value in sorted(self.imbalanceDict.items(),key=lambda x: x[1])]
        n = classes[0][1]
        clsDict = collections.defaultdict(list)
        minority = classes[0][0]
        clsDict[classes[0][0]] = [(self.X[self.y==minority].copy(),self.y[self.y==minority].copy())]
        for cls, size in classes[1:]:
            XClass = self.X[self.y==cls].copy()
            yClass = self.y[self.y==cls].copy()
            ibr = size//n
            while XClass.shape[0]>=n:
                clsDict[cls].append((XClass[:n,:],yClass[:n]))
                XClass,yClass = XClass[n:,:],yClass[n:]
            res = n-XClass.shape[0]
            if res > 0:
                m = np.random.randint(0, n * ibr, res)
                clsDict[cls].append((np.concatenate((XClass,self.X[self.y==cls][m]),axis=0),np.concatenate((yClass,self.y[self.y==cls][m]),axis=None)))
        lenList = sorted([(key,len(value)) for key, value in clsDict.items()],key = lambda x: x[1])
        majority = classes[-1][0]
        maxN = lenList[-1][1]
        for i in range(maxN):
            seq = [clsDict[key][length % i - 1] for key, length in lenList[1:-1]] +[clsDict[majority][i]] + clsDict[minority]
            self.dataLoader.append((shuffle(np.concatenate([x[0] for x in seq],axis=0),np.concatenate([x[1] for x in seq],axis=0))))

    def fit(self,clfs):
        self.clfs = clfs
        self.clfs = [self.clfs[i].fit(self.dataLoader[i][0],self.dataLoader[i][1]) for i in range(len(self.dataLoader))]

    def initializeWeights(self):
        self.weights = [1/len(self.dataLoader) for _ in range(len(self.dataLoader))]

    def predict(self):
        if self.dfProb.empty:
            n = len(self.clfs)
            for i in range(n):
                self.dfProb[f'Classifier {i}'] = self.clfs[i].predict_proba(self.XTest)[:, 1]
        else:
            self.dfProb.drop(['rawProb','prob','pred'],axis=1,inplace=True)
        self.weights = np.array(self.weights)
        self.dfProb['rawProb'] = np.dot(self.dfProb.values, self.weights)
        self.dfProb['prob'] = self.dfProb['rawProb'].apply(lambda x: 1 if x >= 1 else (0 if x <= 0 else x))
        self.dfProb['pred'] = self.dfProb['prob'].apply(lambda x: x > 0.5)
        self.yPred = self.dfProb['pred'].values
        self.yProb = self.dfProb['prob'].values

    def optimizeWeights(self,method='linear'):
        n = len(self.clfs)
        for i in range(n):
            self.dfOptimize[f'Classifier {i}'] = self.clfs[i].predict_proba(self.X)[:, 1]
        if method == 'linear':
            optimizer = LinearRegression(fit_intercept=False)
            optimizer.fit(self.dfOptimize.values, self.y)
            self.weights = optimizer.coef_/optimizer.coef_.sum()
        else:
            if method=='gradient':
                optimizer = GradientBoostingRegressor()
            if method == 'ada':
                optimizer = AdaBoostRegressor()
            optimizer.fit(self.dfOptimize.values,self.y)
            self.weights = optimizer.feature_importances_/optimizer.feature_importances_.sum()

    def evaluate(self):
        self.accuracy = accuracy_score(self.yTest,self.yPred)
        self.confusionMatrix = confusion_matrix(self.yTest,self.yPred)
        print(self.accuracy)
        print(self.confusionMatrix)

    def plot(self):
        # pr/rc against thr
        self.precision,self.recall,self.threshold = precision_recall_curve(self.yTest,self.yProb)
        pr,rc,thr = self.precision,self.recall,self.threshold
        pr,rc = pr[:-1],rc[:-1]

        fig, ax = plt.subplots()
        precision, = ax.plot(thr, pr, color='steelblue', label='Precision')
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Precision')
        ax.set_title(f'Area Under PR/RC Curve: {auc(thr, np.minimum(pr, rc))}')
        ax2 = ax.twinx()
        recall, = ax2.plot(thr, rc, color='darkorange', label='Recall')
        ax2.set_ylabel('Recall')
        plt.legend(handles=[precision, recall])
        plt.show()
        plt.close()

        # ROC Curve
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

        # pr against rc curve
        plt.figure()
        lw = 2
        plt.plot(self.recall, self.precision, color="steelblue", lw=lw)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Average Precision %0.4f" % average_precision_score(self.yTest,self.yProb))
        plt.show()
        plt.close()

    def importances(self):
        for i in range(len(self.clfs)):
            self.dfImportances[f'Importances {i}'] = self.clfs[i].feature_importances_
        importances = np.dot(self.dfImportances.values,self.weights)
        self.featureImportances['Features'] = self.features
        self.featureImportances['Importances'] = minmax_scale(importances,(0,100))
        self.featureImportances.sort_values(by='Importances',ascending=False,inplace=True)
        return self.featureImportances