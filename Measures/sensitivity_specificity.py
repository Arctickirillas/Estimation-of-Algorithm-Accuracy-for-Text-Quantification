__author__ = 'Kirill Rudakov'

from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve

def SensitivityAndSpecificity(pred,real):
    tp,tn,fp,fn = 0,0,0,0

    for i,est in enumerate(pred):
        if est == real[i] and est == 1.:
            tp += 1.
        elif est == real[i] and est == 0:
            tn += 1.
        elif est != real[i] and est==1:
            fp += 1.
        elif est != real[i] and est==0:
            fn += 1.
    # tpr -- tnr
    return tp/(tp+fn),tn/(tn+fp)

def getROC_Curve(pred,real):
    fpr, tpr, thresholds = (roc_curve(pred,real))
    return fpr[1],tpr[1]
