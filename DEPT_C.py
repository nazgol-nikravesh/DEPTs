import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
import rates_measurement as rm


class DEC:
    def __init__(self,X,y,X_test,y_test,parameters,clf, goal):
        self.X = X
        self.y=y
        self.X_test=X_test
        self.y_test=y_test
        self.parameters=parameters
        self.clf=clf
        self.goal=goal
        self.parameter_candidate_pool=[[1.0,0.1],[1.0,0.9],[0.8,0.2]] #F and Cr

    def DEPT_C(self, NP, life):
         Population = []
         for i in range(0, NP):
             cromosome = []
             for para in self.parameters:
                if para[0] == 'b':
                    cromosome.append(bool(random.getrandbits(1)))
                elif para[2] == 1:
                    cromosome.append(random.randrange(para[0], para[1]))
                elif para[2] == 2:
                    cromosome.append(random.uniform(para[0], para[1]))
             Population.append(cromosome)

         PopScrs = self.PopulationScore(Population)
         if self.goal[1]=='+':
            Sbest = Population[PopScrs.index(max(PopScrs))]
         else:
             Sbest = Population[PopScrs.index(min(PopScrs))]

         l = 0
         while life > 0 :#or max(PopScrs)==1.0 and l<=100

             l += 1
             NewGeneration = []
             altPopScrs = []
             for i in range(0, NP):
                 new,scr=self.ChooseBest(Population[i], Population, i)
                 NewGeneration.append(new)
                 altPopScrs.append(scr)

             Population = NewGeneration
             if self.goal[1] == '+':
                bestOld = max(PopScrs)
                bestNew = max(PopScrs)
             else:
                 bestOld = min(PopScrs)
                 bestNew = min(PopScrs)
             sumOld = sum(PopScrs)
             PopScrs = altPopScrs
             sumNew = sum(PopScrs)

             if (bestOld >= bestNew and sumOld >= sumNew and self.goal[1] == '+') or (bestOld <= bestNew and sumOld <= sumNew and self.goal[1] == '-'):
                 life -= 1

             if self.goal[1] == '+':
                Sbest = Population[PopScrs.index(max(PopScrs))]
             else:
                 Sbest = Population[PopScrs.index(min(PopScrs))]

             # print(PopScrs, '-*-**-*-*-*-*-*-*-*-*-*')
         # print(max(PopScrs), Sbest)
         # if self.goal[1] == '+':
         #    print(Sbest,max(PopScrs), 'number of evaluations:', l)
         # else:
         #     print(Sbest, min(PopScrs), 'number of evaluations:', l)
         return Sbest,l

    def PopulationScore(self,Population):
        PopScores =[]
        for i in range(0,len(Population)):
            PopScores.append(self.Score(Population[i]))

        return PopScores

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

    def ChooseBest(self,old, pop, i):

        r = range(0,len(pop))
        i1, i2, i3 ,i4,i5= random.sample(r, 5)
        c=0

        while c ==0 and len(pop)>5:
            if i1 ==i or i2==i or i3==i or i4==i or i5==i:
                i1, i2, i3 , i4, i5= random.sample(r, 5)
            else:
                c +=1

        U1 = self.rand_1_bin(old,pop[i1], pop[i2], pop[i3])
        U2 = self.rand_2_bin(old,pop[i1], pop[i2], pop[i3], pop[i4],pop[i5])
        U3 = self.currentToRand_1(old,pop[i1], pop[i2], pop[i3])
        SU1=self.Score(U1)
        SU2 = self.Score(U2)
        SU3 = self.Score(U3)
        Sold = self.Score(old)
        if (SU1>=SU2 and SU1>=SU3 and SU1>=Sold and self.goal[1]=='+') or (SU1<=SU2 and SU1<=SU3 and SU1<=Sold and self.goal[1]=='-'):
            return U1,SU1
        elif (SU2>=SU1 and SU2>=SU3 and SU2>=Sold and self.goal[1]=='+') or (SU2<=SU1 and SU2<=SU3 and SU2<=Sold and self.goal[1]=='-'):
            return U2, SU2
        elif (SU3>=SU1 and SU3>=SU2 and SU3>=Sold and self.goal[1]=='+') or (SU3<=SU1 and SU3<=SU2 and SU3<=Sold and self.goal[1]=='-'):
            return U3, SU3
        else:
            return old, Sold


    def rand_1_bin(self,old,Xr1, Xr2, Xr3):
        pcp = random.randrange(0, len(self.parameter_candidate_pool))
        F=self.parameter_candidate_pool[pcp][0]
        Cr=self.parameter_candidate_pool[pcp][1]
        U1=[]
        for i in range(0, len(old)):
            if Cr > random.uniform(0, 1) or i==random.randrange(0, len(old)):
                if self.parameters[i][0] == 'b':
                    U1.append(not old[i])
                else:
                    new = Xr1[i] + F * (Xr2[i] - Xr3[i])
                    if new < self.parameters[i][0]:
                        U1.append(self.parameters[i][0])
                    elif new > self.parameters[i][1]:
                        U1.append(self.parameters[i][1])
                    elif self.parameters[i][2] == 1:
                        U1.append(int(new))
                    else:
                        U1.append(new)
            else:
                U1.append(old[i])
        return U1

    def rand_2_bin(self,old,Xr1, Xr2, Xr3, Xr4, Xr5):
        pcp = random.randrange(0, len(self.parameter_candidate_pool))
        F=self.parameter_candidate_pool[pcp][0]
        Cr=self.parameter_candidate_pool[pcp][1]
        U2=[]
        for i in range(0, len(old)):
            if Cr > random.uniform(0, 1) or i==random.randrange(0, len(old)):
                if self.parameters[i][0] == 'b':
                    U2.append(not old[i])
                else:
                    new = Xr1[i] + random.uniform(0, 1) * (Xr2[i] - Xr3[i]) + F *(Xr4[i] - Xr5[i])
                    if new < self.parameters[i][0]:
                        U2.append(self.parameters[i][0])
                    elif new > self.parameters[i][1]:
                        U2.append(self.parameters[i][1])
                    elif self.parameters[i][2] == 1:
                        U2.append(int(new))
                    else:
                        U2.append(new)
            else:
                U2.append(old[i])
        return U2

    def currentToRand_1(self,old,Xr1, Xr2, Xr3):
        pcp = random.randrange(0, len(self.parameter_candidate_pool))
        F = self.parameter_candidate_pool[pcp][0]
        U3 = []
        for i in range(0, len(old)):
                if self.parameters[i][0] == 'b':
                    U3.append(not old[i])
                else:
                    new = old[i] + random.uniform(0, 1) * (Xr1[i] - old[i]) + F * (Xr2[i] - Xr3[i])
                    if new < self.parameters[i][0]:
                        U3.append(self.parameters[i][0])
                    elif new > self.parameters[i][1]:
                        U3.append(self.parameters[i][1])
                    elif self.parameters[i][2] == 1:
                        U3.append(int(new))
                    else:
                        U3.append(new)

        return U3


