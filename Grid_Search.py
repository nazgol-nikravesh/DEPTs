import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
import rates_measurement as rm
import time

class Grid:
    def __init__(self,X,y,X_test,y_test,grid_params,clf, goal):
        self.X = X
        self.y=y
        self.X_test=X_test
        self.y_test=y_test
        self.grid_params=grid_params
        self.clf=clf
        self.goal=goal

    def Grid_search(self, NP, life,start_time):

         l = 0
         c=0
         better=self.grid_params[c]
         while c < len(self.grid_params)-1 and time.time()-start_time< 1800:#or max(PopScrs)==1.0 and l<=100
             print(time.time()-start_time,len(self.grid_params),c)
             l += 1
             c+=1
             worse=better
             better=self.grid_params[c]
             if (self.Score(worse)>self.Score(better) and self.goal[1] == '+') or (self.Score(worse)<self.Score(better) and self.goal[1] == '-'):
                better=worse




         return better,l



    def Score(self, candidate):
        # print(candidate)
        if self.clf == 1:
            # clf = RandomForestClassifier(max_features=candidate[0], max_leaf_nodes=candidate[1],
            #                              min_samples_split=candidate[2], min_samples_leaf=candidate[3],
            #                              n_estimators=candidate[4])
            clf= RandomForestClassifier()
            clf.set_params(**candidate)
            clf.fit(self.X, self.y)
            # pred = clf.predict(self.X_test)
            # print(metrics.f1_score(self.y_test, pred, zero_division=1))
            # print(metrics.precision_score(self.y_test, pred, average='binary'))
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >= 0.5).astype('int')
        elif self.clf == 2:
            clf = DecisionTreeClassifier()
            clf.set_params(**candidate)
            clf.fit(self.X, self.y)
            # pred = clf.predict(self.X_test)
            # print(metrics.f1_score(self.y_test, pred, zero_division=1))
            # print(metrics.precision_score(self.y_test, pred, average='binary'))
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >=  0.5).astype('int')
        elif self.clf == 3:
            # if candidate[1] == False:
            #     weights = 'uniform'
            # elif candidate[1] == True:
            #     weights = 'distance'
            clf = KNeighborsClassifier()
            clf.set_params(**candidate)
            clf.fit(self.X, self.y)
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >=  0.5).astype('int')
        elif self.clf == 4:
            # if candidate[1] == False:
            #     kernel = 'rbf'
            # elif candidate[1] == True:
            #     kernel = 'sigmoid'
            clf = SVC(probability=True)
            clf.set_params(**candidate)
            clf.fit(self.X, self.y)
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >=  0.5).astype('int')
        if self.goal[0] == 'precision':
            return metrics.precision_score(self.y_test, pred, average='binary')
        elif self.goal[0] == 'F1-measure':
            return metrics.f1_score(self.y_test, pred, zero_division=1)
        elif self.goal[0] == 'Recall':
            return metrics.recall_score(self.y_test, pred, zero_division=1)
        elif self.goal[0] == 'GM':
            return rm.G_measure(self.y_test, pred)
        elif self.goal[0] == 'AUC':
            return metrics.roc_auc_score(self.y_test, pred)

        elif self.goal[0] == 'PF':
            return rm.false_positive_rate(self.y_test, pred)
        # elif self.goal=='IFA':
        #     return metrics.f1_score(self.y_test, pred, zero_division=1)
        elif self.goal[0] == 'Brier':
            return metrics.brier_score_loss(self.y_test, pred)
        elif self.goal[0] == 'D2H':
            return rm.D2H(self.y_test, pred)




