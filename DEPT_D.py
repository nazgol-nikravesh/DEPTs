import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import rates_measurement as rm


class DEPT_D:
    def __init__(self,X,y,X_test,y_test,parameters,clf, goal):
        self.X = X
        self.y = y
        self.X_test=X_test
        self.y_test=y_test
        self.parameters=parameters
        self.clf=clf
        self.goal = goal

    def DPLDE(self,NP,f,cr,life):

        # print(self.parameters)
        Population = []
        for i in range(0,NP):
            cromosome=[]
            for para in self.parameters:
                if para[0]=='b':
                    cromosome.append(bool(random.getrandbits(1)))
                elif para[2]==1:
                    cromosome.append(random.randrange(para[0],para[1]))
                elif para[2]==2:
                    cromosome.append(random.uniform(para[0],para[1]))
            Population.append(cromosome)

        PopScrs = self.PopulationScore(Population)
        Sbest = Population[PopScrs.index(max(PopScrs))]

        # while life>0 :
        T=20
        t=1
        l = 0
        while life>0:
            l += 1
            BFI=self.BFI(Population,PopScrs)
            # print(BFI)
            # print(PopScrs)
            BDI=self.BDI(BFI)
            # w2= pow((5-life)/5, 2)
            w2 = pow(t / 5, 2)
            NewGeneration = []
            S =[]
            altPopScrs = []
            alt2PopScrs = []
            for i in range(0,NP):
                fsum=0
                for j in range(0,NP):
                    fsum +=BFI[j][1]
                fmean= fsum/NP
                if fmean-BFI[NP-1][1]==0:
                    w1=100000
                else:
                    w1=(BFI[i][1]-BFI[NP-1][1])/(fmean-BFI[NP-1][1])
                # print(BFI[i][1],BFI[NP-1][1],fmean,BFI[NP-1][1],'nnaaaaaaaaaaaaa3224223444444444444444')
                S.append(self.Extrapolate(BFI[i][0], BFI ,cr,f,i,BDI,w1,w2,NP))
                altPopScrs.append(self.Score(S[i]))
                if (altPopScrs[i] >= BFI[i][1] and self.goal[1] == '+') or (altPopScrs[i] <= BFI[i][1] and self.goal[1] == '-'):
                    NewGeneration.append(S[i])
                    alt2PopScrs.append(altPopScrs[i])
                else:
                    NewGeneration.append(BFI[i][0])
                    alt2PopScrs.append(BFI[i][1])
                # print(self.Score(S[i]))
                # print(PopScrs[i])
                # print('*****')


            Population=NewGeneration
            if self.goal[1] == '+':
                bestOld = max(PopScrs)
                bestNew = max(PopScrs)
            else:
                bestOld = min(PopScrs)
                bestNew = min(PopScrs)
            sumOld=sum(PopScrs)
            # print('-*-**-*-*-*////////////////////////////////////////////////////////-*-*-*-*-*-*')


            PopScrs = alt2PopScrs
            sumNew=sum(PopScrs)

            if (bestOld >= bestNew and sumOld >= sumNew and self.goal[1] == '+') or (bestOld <= bestNew and sumOld <= sumNew and self.goal[1] == '-'):
                life -= 1
                t +=1

        if self.goal[1] == '+':
            Sbest = Population[PopScrs.index(max(PopScrs))]
        else:
            Sbest = Population[PopScrs.index(min(PopScrs))]
        return Sbest,l

    def Score(self, candidate):
        # print(candidate)
        if self.clf == 1:
            clf = RandomForestClassifier(max_features=candidate[0], max_leaf_nodes=candidate[1],
                                         min_samples_split=candidate[2], min_samples_leaf=candidate[3],
                                         n_estimators=candidate[4])

            clf.fit(self.X, self.y)
            # pred = clf.predict(self.X_test)
            # print(metrics.f1_score(self.y_test, pred, zero_division=1))
            # print(metrics.precision_score(self.y_test, pred, average='binary'))
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >= candidate[5]).astype('int')
        elif self.clf == 2:
            clf = DecisionTreeClassifier(max_features=candidate[0], min_samples_split=candidate[1]
                                         , min_samples_leaf=candidate[2], max_depth=candidate[3])
            clf.fit(self.X, self.y)
            # pred = clf.predict(self.X_test)
            # print(metrics.f1_score(self.y_test, pred, zero_division=1))
            # print(metrics.precision_score(self.y_test, pred, average='binary'))
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >= candidate[4]).astype('int')
        elif self.clf == 3:
            if candidate[1] == False:
                weights = 'uniform'
            elif candidate[1] == True:
                weights = 'distance'
            clf = KNeighborsClassifier(n_neighbors=candidate[0], weights=weights)
            clf.fit(self.X, self.y)
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >= candidate[2]).astype('int')
        elif self.clf == 4:
            if candidate[1] == False:
                kernel = 'rbf'
            elif candidate[1] == True:
                kernel = 'sigmoid'
            clf =SVC(probability=True,C=candidate[0], kernel=kernel, coef0=candidate[2], gamma=candidate[3])
            clf.fit(self.X, self.y)
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >= candidate[4]).astype('int')

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

    def PopulationScore(self,Population):
        PopScores =[]
        for i in range(0,len(Population)):
            PopScores.append(self.Score(Population[i]))

        return PopScores

    def Extrapolate(self,old, BFI,cr,f, j, BDI,w1,w2,NP):
        if j != NP - 1:
            r1 = random.randrange(j+ 1, NP)
        else:
            r1 = j
        xBFI = BFI[r1][0]

        k = 0
        for bdi in BDI:
            if bdi[2] == j:
                break
            k += 1
        if k != NP - 1:
            r2 = random.randrange(k + 1, NP)
        else:
            r2 = k
        xBDI = BDI[r2][0]

        xBest=BFI[NP-1][0]
        newf = []
        for i in range(0,len(old)):
            if cr>=random.uniform(0, 1) or random.randrange(0,len(old))==i:
                if self.parameters[i][0]=='b':
                    newf.append(not old[i])
                else:
                    new= w1*f*(xBFI[i]-old[i]) + w2*f*(xBDI[i]-old[i]) + f*(xBest[i]-old[i]) # DEPT_D1
                    # new = w1 * f * (xBFI[i] - old[i]) + w2 * f * (xBDI[i] - old[i]) # DEPT_D2

                    if new< self.parameters[i][0]:
                        new=self.parameters[i][0]
                    elif new> self.parameters[i][1]:
                        new=self.parameters[i][1]
                    if self.parameters[i][2] == 1:
                        newf.append(int(new))
                    else:
                        newf.append(new)
            else:
                newf.append(old[i])
        return newf

    def BFI(self,Population,scores):
        SortedPopScores =[]

        for i in range(0,len(Population)):
            PopScore = []
            PopScore.append(Population[i])
            PopScore.append(scores[i])
            SortedPopScores.append(PopScore)

        if self.goal[1] == '+':
            SortedPopScores.sort(key=lambda x: x[1])
        else:
            SortedPopScores.sort(key=lambda x: x[1],reverse=True)


        return SortedPopScores

    def BDI(self, PopulationBFI):

        d=np.zeros((len(PopulationBFI),len(PopulationBFI)))
        PopDiversityScores=[]
        for i in range(0, len(PopulationBFI)):
            dMean = []
            dSumXi = 0
            for j in range(0, len(PopulationBFI)):
                sum=0
                for k in range(0, len(PopulationBFI[i])):
                    sum += pow(PopulationBFI[i][0][k]-PopulationBFI[j][0][k], 2)
                d[i][j]= pow(sum,0.5)
                dSumXi +=d[i][j]
            dMean.append(PopulationBFI[i][0])
            dMean.append(dSumXi/(len(PopulationBFI)-1))
            dMean.append(i)
            PopDiversityScores.append(dMean)
        if self.goal[1] == '+':
            PopDiversityScores.sort(key=lambda x: x[1])
        else:
            PopDiversityScores.sort(key=lambda x: x[1],reverse=True)


        return PopDiversityScores
