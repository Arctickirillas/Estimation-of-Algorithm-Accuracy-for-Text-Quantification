__author__ = 'Kirill Rudakov'

import pandas as pd
from Measures.sensitivity_specificity import  getROC_Curve
from generate_data import generate_data
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier  as KNC
from Measures.KLD import kld
from QuantificationAlgorithms.ClassifyAndCount import CC,PCC,ACC,PACC

if __name__ == '__main__':
    X,y = generate_data(10000,20)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=15)
    print(pd.Series(y).value_counts())


    clf = KNC(n_neighbors=7)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    fpr,tpr = getROC_Curve(clf.predict(X_train),y_train)

    cc_real = CC(y_test)

    cc = CC(predictions)
    pcc = PCC(predictions,clf.predict_proba(X_test))
    acc = ACC(predictions,fpr,tpr)
    pacc = PACC(predictions,clf.predict_proba(X_test))

    print(kld(cc_real,cc))
    print(kld(cc_real,pcc))
    print(kld(cc_real,acc))
    print(kld(cc_real,pacc))


    # Интересно посмотреть с качеством классификаторов
    # clf = KNC(n_neighbors=7)
    # cv = KFold(len(y), 5, shuffle=True)
    # scores = cross_val_score(clf, X, y, cv=cv)
    # print(scores.mean())
