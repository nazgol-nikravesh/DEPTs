from sklearn import metrics
import math

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN

def false_positive_rate(y_actual, y_hat):
    TP, FP, TN, FN = perf_measure(y_actual, y_hat)
    if FP+TN==0:
        return 'error 0 devision'
    else:
        return FP/(FP+TN)

def G_measure(y_actual, y_hat):
    PF=false_positive_rate(y_actual, y_hat)
    Recall=metrics.recall_score(y_actual, y_hat, zero_division=1)
    if Recall+(1-PF)==0:
        return 'error 0 devision'
    else:
        return (2*Recall*(1-PF))/(Recall+(1-PF))

def D2H(y_actual, y_hat):
    PF = false_positive_rate(y_actual, y_hat)
    Recall = metrics.recall_score(y_actual, y_hat, zero_division=1)
    return (math.pow((math.pow(1-Recall, 2)+math.pow(0-PF, 2)), 0.5))/math.pow(2, 0.5)