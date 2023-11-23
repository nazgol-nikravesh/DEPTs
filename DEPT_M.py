import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import cauchy
import numpy as np
import math
import rates_measurement as rm

class DEPT_M:
    def __init__(self,X,y,X_test,y_test,parameters,clf, goal):
        self.X = X
        self.y = y
        self.X_test=X_test
        self.y_test=y_test
        self.parameters=parameters
        self.clf=clf
        self.goal=goal


    def mplde(self, NP, f, cr, life):
        w1=0.5
        w2=0.4
        w3=0.1
        # max_FES=1000
        Gmax=100
        G=1
        # FES=0
        Fmw=0.5
        Fmm=0.5
        Fmb=0.5
        Crmw=0.5
        Crmm=0.5
        Crmb=0.5
        PG = []
        for i in range(0,NP):
            cromosome=[]
            for para in self.parameters:
                if para[0]=='b':
                    cromosome.append(bool(random.getrandbits(1)))
                elif para[2]==1:
                    cromosome.append(random.randrange(para[0],para[1]))
                elif para[2]==2:
                    cromosome.append(random.uniform(para[0],para[1]))
            PG.append(cromosome)
        SF=[]
        SCr=[]
        l=0
        while life > 0:
            l+=1
            FCrs = []
            UG=[]
            # print('PGGGGGGGGGGG:',PG)
            PG=self.sortByScores(PG)
            UG,FCrs=self.makeUG(NP,w1,w2,Fmm,Fmw,Fmb,Crmm,Crmw,Crmb,PG)
            # UGsorted=self.sortByScores(UG)
            # FES +=NP
            oldscrs=self.PopulationScore2(PG)
            if self.goal[1] == '+':
                bestOld=max(oldscrs)
            else:
                bestOld=min(oldscrs)

            sumOld=sum(oldscrs)
            for i in range(0,NP):
                # print('UGGGGGGGG:',UG[i])
                UGiScore=self.Score(UG[i])

                if (UGiScore>=PG[i][1] and self.goal[1]=='+') or (UGiScore<=PG[i][1] and self.goal[1]=='-'):
                    new=[]
                    new.append(UG[i])
                    new.append(UGiScore)
                    PG[i]=new
                    SF.append(FCrs[i][0])
                    SCr.append(FCrs[i][1])


            Fmw,Fmm,Fmb,Crmw,Crmm,Crmb=self.makeFandCr(SF,Fmw,Fmm,Fmb,SCr,Crmw,Crmm,Crmb)

            count=0
            if random.uniform(0, 1)<=(G-1)/Gmax:
                count=math.floor(3/100*random.uniform(0, 1)*NP)
                # print('------------count:',count)
            UGsorted = self.sortByScores(UG)

            if self.goal[1]=='+':
                PG.sort(key=lambda x: x[1])
            else:
                PG.sort(key=lambda x: x[1],reverse=True)

            for i in range(0,count):
                PG[i]=UGsorted[NP-i-1]

            G +=1
            newscrs=self.PopulationScore2(PG)
            if self.goal[1]=='+':
                bestNew = max(newscrs)
            else:
                bestNew = min(newscrs)

            sumNew = sum(newscrs)
            if (bestOld >= bestNew and self.goal[1]=='+') or (bestOld <= bestNew and self.goal[1]=='-'): #and sumOld >= sumNew
                life -= 1
            # print('*-*-*-*-*',bestNew,'*-*-*-*-*')
            PG=self.getPopulation(PG)

        PG=self.sortByScores(PG)
        # print(PG[NP-1],'number of evaluations:',l)
        return PG[NP-1][0],l

    def makeFandCr(self,SF,Fmw,Fmm,Fmb,SCr,Crmw,Crmm,Crmb):
        if len(SF) > 0 and sum(SF) > 0:
            wF = 0.8 + 0.2 * random.uniform(0, 1)
            sumFpow2 = 0
            sumF = 0

            for i in range(0, len(SF)):
                sumF += SF[i]
                sumFpow2 += pow(SF[i], 2)

            meanLSF = sumFpow2 / sumF
            Fmw = wF + Fmw + (1 - wF) * meanLSF
            Fmm = wF + Fmm + (1 - wF) * meanLSF
            Fmb = wF + Fmb + (1 - wF) * meanLSF

        else:
            CF = 0.5 * random.uniform(0, 1)
            Fmw = CF * Fmw + (1 - CF) * random.uniform(0, 1)
            Fmm = CF * Fmm + (1 - CF) * random.uniform(0, 1)
            Fmb = CF * Fmb + (1 - CF) * random.uniform(0, 1)


        wCr = 0.5 * random.uniform(0, 1)

        if len(SCr) > 0 and sum(SCr) > 0:
            sumCrpow2 = 0
            sumCr = 0

            for i in range(0, len(SCr)):
                sumCr += SCr[i]
                sumCrpow2 += pow(SCr[i], 2)

            meanLSCr = sumCrpow2 / sumCr
            Crmw = wCr * Crmw + (1 - wCr) * meanLSCr
            Crmm = wCr * Crmm + (1 - wCr) * meanLSCr
            Crmb = wCr * Crmb + (1 - wCr) * meanLSCr
        else:

            Crmw = wCr * Crmw + (1 - wCr) * random.uniform(0, 1)
            Crmm = wCr * Crmm + (1 - wCr) * random.uniform(0, 1)
            Crmb = wCr * Crmb + (1 - wCr) * random.uniform(0, 1)

        return Fmw,Fmm,Fmb,Crmw,Crmm,Crmb

    def makeUG(self,NP,w1,w2,Fmm,Fmw,Fmb,Crmm,Crmw,Crmb,PG):
        FCrs=[]
        UG=[]
        for i in range(0, NP):
            F = 0
            Cr = 0
            frm = 0
            to = 0
            if i + 1 <= int(w1 * NP):
                F = Fmw
                Cr = Crmw
                frm = 0
                to = int(w1 * NP)

            elif i + 1 <= int((w1 + w2) * NP):
                F = Fmm
                Cr = Crmm
                frm = int(w1 * NP)
                to = int((w1 + w2) * NP)

            else:
                F = Fmb
                Cr = Crmb
                frm = int((w1 + w2) * NP)
                to = NP

            Fi = cauchy.rvs(loc=F, scale=0.1, size=1, random_state=None)[0]
            if Fi<0:
                Fi=0
            elif Fi>1:
                Fi=1
            Cri = np.random.normal(Cr, 0.1, 1)[0]
            if Cri<0:
                Cri=0
            elif Cri>1:
                Cri=1
            if i + 1 == to:
                Xrbest = PG[to - 2][0]
            else:
                Xrbest = PG[to - 1][0]

            r = range(frm, to)

            r1, r2, r3, r4 = random.sample(r, 4)
            c = 0
            d = to-frm
            while c == 0 and d>4:
                if r1 == i or r2 == i or r3 == i or r4 == i:
                    r1, r2, r3, r4 = random.sample(r, 4)
                else:
                    c += 1
            FCrtemp = []
            FCrtemp.append(Fi)
            FCrtemp.append(Cri)
            FCrs.append(FCrtemp)
            Ui = self.mutationAndCrossOver(PG[i][0], Fi, Xrbest, PG[r1][0], PG[r2][0], PG[r3][0], PG[r4][0], Cri)
            UG.append(Ui)
        return UG,FCrs

    def mutationAndCrossOver(self,Xi, Fi, Xrbest, Xr1, Xr2, Xr3, Xr4,Cri):

        Ui=[]
        for i in range(0, len(Xi)):
            if Cri >= random.uniform(0, 1) or random.randrange(0, len(Xi)) == i:
                if self.parameters[i][0] == 'b':
                    Ui.append(not Xi[i])
                else:
                    new = Xi[i] +Fi*(Xr1[i]-Xi[i])+ Fi*(Xr3[i]-Xr4[i])+ Fi*(Xrbest[i]-Xi[i]) #DEPT_M1
                    # new = Xi[i] +Fi*(Xr1[i]-Xi[i])+ Fi*(Xr3[i]-Xr4[i]) #DEPT_M2

                    if new < self.parameters[i][0]:
                        Ui.append(self.parameters[i][0])
                    elif new > self.parameters[i][1]:
                        Ui.append(self.parameters[i][1])
                    elif self.parameters[i][2] == 1:
                        Ui.append(int(new))
                    else:
                        Ui.append(float(new))

                # Ui=Vi
            else:
                if self.parameters[i][0] == 'b':
                    Ui.append(Xi[i])
                elif self.parameters[i][2] == 1:
                    Ui.append(int(Xi[i]))
                elif self.parameters[i][2] == 2:
                    Ui.append(float(Xi[i]))
                else:
                    Ui.append(Xi[i])

                # Ui=Xi


        return Ui


    def sortByScores(self, Population):
        SortedPopScores = []

        for i in range(0, len(Population)):
            PopScore = []
            PopScore.append(Population[i])
            PopScore.append(self.Score(Population[i]))
            SortedPopScores.append(PopScore)

        if self.goal[1]=='+':
            SortedPopScores.sort(key=lambda x: x[1])
        else:
            SortedPopScores.sort(key=lambda x: x[1],reverse=True)


        return SortedPopScores

    def Score(self, candidate):
        # print(candidate)
        if self.clf==1:
            clf = RandomForestClassifier(max_features=candidate[0], max_leaf_nodes=candidate[1],
                                         min_samples_split=candidate[2], min_samples_leaf=candidate[3],
                                         n_estimators=candidate[4])

            clf.fit(self.X, self.y)
            # pred = clf.predict(self.X_test)
            # print(metrics.f1_score(self.y_test, pred, zero_division=1))
            # print(metrics.precision_score(self.y_test, pred, average='binary'))
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >= candidate[5]).astype('int')
        elif self.clf==2:
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
            clf = SVC(probability=True,C=candidate[0], kernel=kernel, coef0=candidate[2], gamma=candidate[3])
            clf.fit(self.X, self.y)
            predicted_proba = clf.predict_proba(self.X_test)
            pred = (predicted_proba[:, 1] >= candidate[4]).astype('int')
        if self.goal[0]=='precision':
            return metrics.precision_score(self.y_test, pred, average='binary')
        elif self.goal[0]=='F1-measure':
            return metrics.f1_score(self.y_test, pred, zero_division=1)
        elif self.goal[0]=='Recall':
            return metrics.recall_score(self.y_test, pred, zero_division=1)
        elif self.goal[0]=='GM':
            return rm.G_measure(self.y_test, pred)
        elif self.goal[0]=='AUC':
            return metrics.roc_auc_score(self.y_test, pred)

        elif self.goal[0]=='PF':
            return rm.false_positive_rate(self.y_test,pred)
        # elif self.goal=='IFA':
        #     return metrics.f1_score(self.y_test, pred, zero_division=1)
        elif self.goal[0]=='Brier':
            return metrics.brier_score_loss(self.y_test, pred)
        elif self.goal[0]=='D2H':
            return rm.D2H(self.y_test, pred)





    def PopulationScore(self, Population):
        PopScores = []
        for i in range(0, len(Population)):
            PopScores.append(self.Score(Population[i]))

        return PopScores

    def PopulationScore2(self, Population):
        PopScores = []
        for i in range(0, len(Population)):
            PopScores.append(Population[i][1])
        return PopScores
    def getPopulation(self,PS):
        Pop = []
        for i in range(0, len(PS)):
            Pop.append(PS[i][0])
        return Pop