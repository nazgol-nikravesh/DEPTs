import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import rates_measurement as rm



class DE:
    def __init__(self,X,y,X_test,y_test,parameters,clf, goal):
        self.X = X
        self.y = y
        self.X_test=X_test
        self.y_test=y_test
        self.parameters=parameters
        self.clf=clf
        self.goal=goal

    def DE(self,NP,f,cr,life):

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

        l = 0
        while life > 0:
            l += 1
            NewGeneration = []
            S =[]
            altPopScrs = []
            alt2PopScrs = []
            for i in range(0,NP):
                S.append(self.Extrapolate(Population[i], Population ,cr,f,i))
                altPopScrs.append(self.Score(S[i]))
                if (altPopScrs[i] > PopScrs[i] and  self.goal[1] == '+') or (altPopScrs[i] < PopScrs[i] and  self.goal[1] == '-'):
                    NewGeneration.append(S[i])
                    alt2PopScrs.append(altPopScrs[i])
                else:
                    NewGeneration.append(Population[i])
                    alt2PopScrs.append(PopScrs[i])
                # print(self.Score(S[i]))
                # print(PopScrs[i])
                # print('*****')


            Population=NewGeneration
            if self.goal[1] == '+':
                bestOld=max(PopScrs)
                bestNew = max(PopScrs)
            else:
                bestOld=min(PopScrs)
                bestNew = min(PopScrs)
            sumOld=sum(PopScrs)
            PopScrs = alt2PopScrs
            sumNew=sum(PopScrs)

            if (bestOld >= bestNew and sumOld >= sumNew and self.goal[1] == '+') or (bestOld <= bestNew and sumOld <= sumNew and self.goal[1] == '-'):
                # print('ooooooooooooooo')
                life -= 1

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

        elif self.goal[0] == 'Brier':
            return metrics.brier_score_loss(self.y_test, pred)
        elif self.goal[0] == 'D2H':
            return rm.D2H(self.y_test, pred)

    def PopulationScore(self,Population):
        PopScores =[]
        for i in range(0,len(Population)):
            PopScores.append(self.Score(Population[i]))

        return PopScores

    def Extrapolate(self,old, pop , cr, f, i):
        r = range(0,len(pop))
        ia, ib, ic= random.sample(r, 3)
        c=0
        while c !=0:
            if ia ==i or ib==i or ic==i:
                ia, ib, ic = random.sample(r, 3)
            else:
                c +=1
        a=pop[ia]
        b=pop[ib]
        c=pop[ic]
        newf = []
        for i in range(0,len(old)):
            if cr < random.uniform(0, 1):
                newf.append(old[i])
            else:
                if self.parameters[i][0]=='b':
                    newf.append(not old[i])
                else:
                    new=a[i]+f*(b[i]-c[i])
                    if new< self.parameters[i][0]:
                        new=self.parameters[i][0]
                    elif new> self.parameters[i][1]:
                        new=self.parameters[i][1]
                    if self.parameters[i][2] == 1:
                        newf.append(int(new))
                    else:
                        newf.append(new)
        return newf

