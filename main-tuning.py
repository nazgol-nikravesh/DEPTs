import DE as TDE
import DEPT_C as CDE
import DEPT_D
import DEPT_M
import Random_Search
import Grid_Search
import rates_measurement as rm
import utilities as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
import warnings
import timeit
import xlsxwriter


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    NP = 10
    f = 0.75
    cr = 0.8
    life = 5
    # DATASETS

    DS=[["ant v0  ","ant-1.3","ant-1.4","ant-1.5"],["ant v1  ","ant-1.4","ant-1.5","ant-1.6"],["ant v2  ","ant-1.5","ant-1.6","ant-1.7"]
        ,["camel v0  ","camel-1.0","camel-1.2","camel-1.4"],["camel v1  ","camel-1.2","camel-1.4","camel-1.6"],
        ["ivy  ","ivy-1.1","ivy-1.4","ivy-2.0"]
        ,["jedit v0  ","jedit-3.2","jedit-4.0","jedit-4.1"],["jedit v1  ","jedit-4.0","jedit-4.1","jedit-4.2"],
        ["jedit v2  ","jedit-4.1","jedit-4.2","jedit-4.3"]
        ,["log4j  ","log4j-1.0","log4j-1.1","log4j-1.2"],
        ["lucene  ","lucene-2.0","lucene-2.2","lucene-2.4"],
        ["poi v0  ","poi-1.5","poi-2.0","poi-2.5"],["poi v1  ","poi-2.0","poi-2.5","poi-3.0"]
        ,["synapse  ","synapse-1.0","synapse-1.1","synapse-1.2"],
        ["velocity  ","velocity-1.4","velocity-1.5","velocity-1.6"]
        ,["xerces v0  ","xerces-init","xerces-1.2","xerces-1.3"],["xerces v1  ","xerces-1.2","xerces-1.3","xerces-1.4"]]

    goals = [['AUC', '+'], ['Brier', '-'],['precision','+'],['F1-measure','+'],['Recall','+'],['GM','+'],['AUC','+'],['PF','-']]
    clfs_type=['RF','CART','KNN','SVM']

    for trial in range(1,2):
        print('trial: ',str(trial))
        for goal in goals:
            print('--> ' + goal[0], goal[1])

            workbook = xlsxwriter.Workbook('trials new/trial'+str(trial)+'/percent/'+goal[0] + str(trial) + '-2.xlsx')
            worksheet= workbook.add_worksheet()
            workbookEval = xlsxwriter.Workbook('trials new/trial'+str(trial)+'/number of evaluations/'+goal[0] +'Eval' + str(trial) + '-2.xlsx')
            worksheetEval = workbookEval.add_worksheet()
            workbookRunTime = xlsxwriter.Workbook('trials new/trial'+str(trial)+'/run time/'+goal[0] +'RunTime' + str(trial) + '-2.xlsx')
            worksheetRunTime = workbookRunTime.add_worksheet()

            worksheet.set_column('A:A', 20)
            worksheet.write('A1', 'datasets')
            worksheet.set_column('B:B', 20)
            worksheet.write('B1', 'KNN')
            worksheet.set_column('C:C', 20)
            worksheet.write('C1', 'KNN-DEPT_D')
            worksheet.set_column('D:D', 20)
            worksheet.write('D1', 'KNN-DE')
            worksheet.set_column('E:E', 20)
            worksheet.write('E1', 'KNN-CoDE')
            worksheet.set_column('F:F', 20)
            worksheet.write('F1', 'KNN-DEPT_M')

            worksheet.set_column('G:G', 20)
            worksheet.write('G1', 'SVM')
            worksheet.set_column('H:H', 20)
            worksheet.write('H1', 'SVM-DEPT_D')
            worksheet.set_column('I:I', 20)
            worksheet.write('I1', 'SVM-DE')
            worksheet.set_column('J:J', 20)
            worksheet.write('J1', 'SVM-CoDE')
            worksheet.set_column('K:K', 20)
            worksheet.write('K1', 'SVM-DEPT_M')

            worksheetEval.set_column('A:A', 20)
            worksheetEval.write('A1', 'datasets')
            worksheetEval.set_column('B:B', 20)
            worksheetEval.write('B1', 'KNN')
            worksheetEval.set_column('C:C', 20)
            worksheetEval.write('C1', 'KNN-DEPT_D')
            worksheetEval.set_column('D:D', 20)
            worksheetEval.write('D1', 'KNN-DE')
            worksheetEval.set_column('E:E', 20)
            worksheetEval.write('E1', 'KNN-CoDE')
            worksheetEval.set_column('F:F', 20)
            worksheetEval.write('F1', 'KNN-DEPT_M')

            worksheetEval.set_column('G:G', 20)
            worksheetEval.write('G1', 'SVM')
            worksheetEval.set_column('H:H', 20)
            worksheetEval.write('H1', 'SVM-DEPT_D')
            worksheetEval.set_column('I:I', 20)
            worksheetEval.write('I1', 'SVM-DE')
            worksheetEval.set_column('J:J', 20)
            worksheetEval.write('J1', 'SVM-CoDE')
            worksheetEval.set_column('K:K', 20)
            worksheetEval.write('K1', 'SVM-DEPT_M')

            worksheetRunTime.set_column('A:A', 20)
            worksheetRunTime.write('A1', 'datasets')
            worksheetRunTime.set_column('B:B', 20)
            worksheetRunTime.write('B1', 'KNN')
            worksheetRunTime.set_column('C:C', 20)
            worksheetRunTime.write('C1', 'KNN-DEPT_D')
            worksheetRunTime.set_column('D:D', 20)
            worksheetRunTime.write('D1', 'KNN-DE')
            worksheetRunTime.set_column('E:E', 20)
            worksheetRunTime.write('E1', 'KNN-CoDE')
            worksheetRunTime.set_column('F:F', 20)
            worksheetRunTime.write('F1', 'KNN-DEPT_M')

            worksheetRunTime.set_column('G:G', 20)
            worksheetRunTime.write('G1', 'SVM')
            worksheetRunTime.set_column('H:H', 20)
            worksheetRunTime.write('H1', 'SVM-DEPT_D')
            worksheetRunTime.set_column('I:I', 20)
            worksheetRunTime.write('I1', 'SVM-DE')
            worksheetRunTime.set_column('J:J', 20)
            worksheetRunTime.write('J1', 'SVM-DEPT_C')
            worksheetRunTime.set_column('K:K', 20)
            worksheetRunTime.write('K1', 'SVM-DEPT_M')

            j = 0
            for ds in DS:
                j +=1
                datasetName=ds[0]
                X, y = ut.read_dataset("datasets/", dataset_name=ds[1])
                X_validation, y_validation = ut.read_dataset("datasets/", dataset_name=ds[2])
                X_test, y_test = ut.read_dataset("datasets/", dataset_name=ds[3])
                # Write some numbers, with row/column notation.
                worksheet.write(j, 0, datasetName)
                worksheetEval.write(j, 0, datasetName)
                worksheetRunTime.write(j, 0, datasetName)


                print('**',datasetName)
                for clf_type in clfs_type:
                    print('===> '+clf_type)
                    if clf_type == 'RF':
                        parameters = [[0.01, 1.00, 2], [2, 50, 1], [2, 20, 1], [1, 20, 1], [50, 150, 1], [0.01, 1.0, 2]]
                        type=1
                    elif clf_type == 'CART':
                        parameters = [[0.01, 1.00, 2], [2, 20, 1], [1, 20, 1],[1, 50, 1], [0.01, 1.0, 2]]
                        type = 2
                    elif clf_type == 'KNN':
                        parameters = [[2, 10, 1], ['b',5], [0.01, 1.0, 2]]
                        type = 3
                    elif clf_type == 'SVM':
                        parameters = [[1.0, 100.0, 2], ['b',5], [0.1, 1.0, 2], [0.1, 1.0, 2], [0.01, 1.0, 2]]
                        type = 4


                    for i in range(0,5):
                                numOfEval=0
                                DE=''
                                if i==0:
                                    start = timeit.default_timer()
                                    dep_m = DEPT_M.DEPT_M(X, y, X_validation, y_validation, parameters, type, goal)
                                    candidate,numOfEval = dep_m.mplde(40, 0.2, 0.8, life)
                                    DE = 'DEPT_M'

                                if i==1:
                                    start = timeit.default_timer()
                                    cde = CDE.DEC(X, y, X_validation, y_validation, parameters, type, goal)
                                    candidate,numOfEval = cde.DEPT_C(NP, life)
                                    DE = 'DEPT_C'
                                elif i==2:
                                    start = timeit.default_timer()
                                    tde = TDE.DE(X, y, X_validation, y_validation, parameters, type, goal)
                                    candidate,numOfEval = tde.DE(NP, f, cr, life)
                                    DE = 'DE'
                                elif i==3:
                                    start = timeit.default_timer()
                                    dplde=DEPT_D.DEPT_D(X, y, X_validation, y_validation, parameters, type, goal)
                                    candidate,numOfEval=dplde.DPLDE(10,0.2,0.8,life)
                                    DE = 'DEPT_D'
                                print(DE)
                                print('Selected optimal parameter values:',candidate)
                                if clf_type=='CART':
                                    k=5
                                    if i!=4:
                                        clf = DecisionTreeClassifier(max_features=candidate[0],min_samples_split=candidate[1]
                                                                     , min_samples_leaf=candidate[2],max_depth=candidate[3])
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= candidate[4]).astype('int')
                                    else:

                                        start = timeit.default_timer()
                                        DE='default'
                                        clf = DecisionTreeClassifier(max_features=None, min_samples_split=2,
                                                                     min_samples_leaf=1,max_depth=None)
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= 0.5).astype('int')
                                elif clf_type=='RF':
                                    k=10
                                    if i!=4:
                                        clf = RandomForestClassifier(max_features=candidate[0], max_leaf_nodes=candidate[1],
                                                                     min_samples_split=candidate[2], min_samples_leaf=candidate[3],
                                                                     n_estimators=candidate[4])
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= candidate[5]).astype('int')


                                    else:
                                        start = timeit.default_timer()
                                        DE = 'default'
                                        clf = RandomForestClassifier(max_features=None, max_leaf_nodes=None,
                                                                     min_samples_split=2, min_samples_leaf=1,
                                                                     n_estimators=100)
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= 0.5).astype('int')
                                elif clf_type=='KNN':
                                    k=5
                                    if i!=4:
                                        if candidate[1]==False:
                                            weights='uniform'
                                        elif candidate[1]==True:
                                            weights='distance'
                                        clf = KNeighborsClassifier(n_neighbors=candidate[0],weights=weights)
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= candidate[2]).astype('int')
                                    else:

                                        start = timeit.default_timer()
                                        DE='default'
                                        clf = KNeighborsClassifier(n_neighbors=5,weights='uniform')
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= 0.5).astype('int')
                                elif clf_type=='SVM':
                                    k=10
                                    if i!=4:
                                        if candidate[1]==False:
                                            kernel='rbf'
                                        elif candidate[1]==True:
                                            kernel='sigmoid'
                                        clf = SVC(probability=True,C=candidate[0],kernel=kernel
                                                                     , coef0=candidate[2],gamma=candidate[3])
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= candidate[4]).astype('int')
                                    else:

                                        start = timeit.default_timer()
                                        DE='default'
                                        clf = SVC(probability=True,C=1.0,kernel='rbf', coef0=0.0,gamma='auto')
                                        clf.fit(X, y)
                                        stop = timeit.default_timer()

                                        predicted_proba = clf.predict_proba(X_test)
                                        pred = (predicted_proba[:, 1] >= 0.5).astype('int')
                                # print(pred)
                                if goal[0]=='precision':
                                    a = metrics.precision_score(y_test, pred, average='binary')
                                elif goal[0]=='F1-measure':
                                    a = metrics.f1_score(y_test, pred, zero_division=1)
                                elif goal[0]=='Recall':
                                    a=metrics.recall_score(y_test, pred, zero_division=1)
                                    # g = rm.false_positive_rate(y_test, pred)
                                elif goal[0] == 'GM':
                                    a=rm.G_measure(y_test, pred)
                                elif goal[0] == 'AUC':
                                    a=metrics.roc_auc_score(y_test, predicted_proba[:, 1])
                                    # print(a)
                                elif goal[0] == 'PF':
                                    a=rm.false_positive_rate(y_test,pred)
                                elif goal[0] == 'Brier':
                                    a=metrics.brier_score_loss(y_test, predicted_proba[:, 1])
                                    # print(a)
                                elif goal[0] == 'D2H':
                                    a=rm.D2H(y_test, pred)

                                worksheet.write(j, k - i, a)
                                worksheetEval.write(j, k - i, numOfEval)
                                worksheetRunTime.write(j, k - i, stop - start)

            workbook.close()
            workbookEval.close()
            workbookRunTime.close()
        print('----------------------end of trial : ',str(trial),'----------------------------')


