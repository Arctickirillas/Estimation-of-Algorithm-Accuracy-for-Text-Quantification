__author__ = 'Kirill Rudakov'
import pandas as pd

def CC(sequence):
    sequence = pd.Series(sequence)
    return [value/sequence.count() for value in sequence.value_counts()]

def PCC(sequence,probabilities):
    sequence = pd.Series(sequence)
    probabilities = pd.DataFrame(probabilities)
    return [sum(probabilities[column_name])/sequence.count() for column_name in probabilities.columns]

def ACC(sequence,fpr,tpr):
    sequence = pd.Series(sequence)
    acc = [(value/sequence.count() - fpr)/(tpr - fpr) for value in sequence.value_counts()]
    return ([cl/sum(acc) for cl in acc])

def PACC(sequence,probabilities):
    sequence = pd.Series(sequence)
    probabilities = pd.DataFrame(probabilities)
    e_fpr,e_tpr = 0,0
    for i,el in enumerate(sequence):
        if el==0:
            e_fpr += probabilities[0].values[i]
        elif el==1:
            e_tpr +=probabilities[1].values[i]
    e_fpr = e_fpr/sequence.value_counts()[0]
    e_tpr = e_tpr/sequence.value_counts()[1]

    pcc  = PCC(sequence,probabilities)
    pacc = [(pcc[i] - e_fpr)/(e_tpr - e_fpr) for i,column_name in enumerate(probabilities.columns)]

    return ([cl/sum(pacc) for cl in pacc])
