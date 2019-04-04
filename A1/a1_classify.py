from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
import numpy as np
from scipy import stats
import argparse
import sys
import os
import csv

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    #print ('TODO')
    total = 0
    cor = 0
    for i, row in enumerate(C):
        for j, ele in enumerate(row):
            total += ele
            if i==j:
                cor += ele
    return cor/total

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    #print ('TODO')
    recalls = []
    for i, row in enumerate(C):
        cor = None
        total = 0
        for j, elem in enumerate(row):
            if i == j:
                cor = elem
            total += elem
        recalls.append(cor/total)
    return recalls

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    #print ('TODO')
    percisions = []
    for column in range(C.shape[1]):
        cor = None
        total = 0
        for j, elem in enumerate(C[:,column]):
            if column == j:
               cor = elem
            total += elem
        percisions.append(cor/total)
    return percisions

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    #print('TODO Section 3.1')
    accuracys = []
    csv_output = []

    npz = np.load(filename)
    feats = npz[npz.files[0]]
    X = feats[:, 0:173]
    y = feats[:, 173]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2)

    print(np.shape(X_train), np.shape(X_test))
    # iBest = 5
    # return (X_train, X_test, y_train, y_test, iBest)

#============================ svc =====================================
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confu = confusion_matrix(y_test, y_pred)

    print(accuracy(confu))
    print(recall(confu))
    print(precision(confu))
    row = []
    row.append(1)
    row.append(accuracy(confu))
    row.extend(recall(confu))
    row.extend(precision(confu))
    row.extend(confu.flatten())
    csv_output.append(row)
    accuracys.append(accuracy(confu))

    # ============================ svc gamma=2 =====================================
    clf = SVC(gamma=2, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confu = confusion_matrix(y_test, y_pred)

    print(accuracy(confu))
    print(recall(confu))
    print(precision(confu))
    row = []
    row.append(2)
    row.append(accuracy(confu))
    row.extend(recall(confu))
    row.extend(precision(confu))
    row.extend(confu.flatten())
    csv_output.append(row)

    accuracys.append(accuracy(confu))


    #=============================random forest =================================

    clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confu = confusion_matrix(y_test, y_pred)

    print(accuracy(confu))
    print(recall(confu))
    print(precision(confu))
    row = []
    row.append(3)
    row.append(accuracy(confu))
    row.extend(recall(confu))
    row.extend(precision(confu))
    row.extend(confu.flatten())
    csv_output.append(row)

    accuracys.append(accuracy(confu))

    # ==================================== mlp =======================================
    clf = MLPClassifier(alpha=0.05)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confu = confusion_matrix(y_test, y_pred)

    print(accuracy(confu))
    print(recall(confu))
    print(precision(confu))
    row = []
    row.append(4)
    row.append(accuracy(confu))
    row.extend(recall(confu))
    row.extend(precision(confu))
    row.extend(confu.flatten())
    csv_output.append(row)

    accuracys.append(accuracy(confu))


    # ========================= adaboost ===============================
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confu = confusion_matrix(y_test, y_pred)

    print(accuracy(confu))
    print(recall(confu))
    print(precision(confu))
    row = []
    row.append(5)
    row.append(accuracy(confu))
    row.extend(recall(confu))
    row.extend(precision(confu))
    row.extend(confu.flatten())
    csv_output.append(row)

    accuracys.append(accuracy(confu))

    print(accuracys)

    #write to csv
    with open('a1_3.1.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        for row in csv_output:
            spamwriter.writerow(row)

    bestAccu = 0
    iBest = None
    for i,acc in enumerate(accuracys):
        if acc > bestAccu:
            bestAccu = acc
            iBest = i
    print("best classifier: " + str(iBest+1))
    return (X_train, X_test, y_train, y_test,iBest+1)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    #print('TODO Section 3.2')
    clf = None
    if iBest == 1:
        clf = LinearSVC()
    elif iBest == 2:
        clf = SVC(gamma=2, max_iter=1000)
    elif iBest == 3:
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif iBest == 4:
        clf = MLPClassifier(alpha=0.05)
    elif iBest == 5:
        clf = AdaBoostClassifier()
    increments = [1000, 5000, 10000, 15000, 20000]
    accuracys = []

    rand_ind = np.random.choice(np.arange(X_train.shape[0]), 1000, replace=False)
    X_1k = X_train[rand_ind]
    y_1k = y_train[rand_ind]

    for incre in increments:
        x_sample = None
        y_sample = None
        if incre == 1000:
            x_sample = X_1k
            y_sample = y_1k
        else:
            rand_ind = np.random.choice(np.arange(X_train.shape[0]), incre, replace=False)
            x_sample = X_train[rand_ind]
            y_sample = y_train[rand_ind]

        clf.fit(x_sample, y_sample)
        y_pred = clf.predict(X_test)
        confu = confusion_matrix(y_test, y_pred)
        accuracys.append(accuracy(confu))
        print(accuracys)


    #write to csv
    with open('a1_3.2.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(accuracys)

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    #print('TODO Section 3.3')

    # ================ 1 ===============================
    ks = [5, 10 ,20, 30, 40, 50]
    p_1k = []
    p_32k = []
    for k in ks:
        selector = SelectKBest(f_classif, k)
        X_new = selector.fit_transform(X_train, y_train)
        pp_index = selector.get_support(indices=True)
        pp = selector.pvalues_
        p_32k_row = [k]
        p_32k_row.extend(pp[pp_index])
        p_32k.append(p_32k_row)

        X_new = selector.fit_transform(X_1k,y_1k)
        pp_index = selector.get_support(indices=True)
        pp = selector.pvalues_
        p_1k_row = [k]
        p_1k_row.extend(pp[pp_index])
        p_1k.append(p_1k_row)

    #print(p_1k)
    #print(p_32k)
    #write 1 to csv
    with open('a1_3.3.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        for row in p_32k:
            spamwriter.writerow(row)

    #============== 2=========================================
    clf = None
    if i == 1:
        clf = LinearSVC()
    elif i == 2:
        clf = SVC(gamma=2, max_iter=1000)
    elif i == 3:
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif i == 4:
        clf = MLPClassifier(alpha=0.05)
    elif i == 5:
        clf = AdaBoostClassifier()
    accuracys = []

    selector = SelectKBest(f_classif, 5)
    #=================== 1k 5 features train ========================
    X_new = selector.fit_transform(X_1k, y_1k)
    x_test_ind = selector.get_support(indices=True)
    x_test_new = X_test[:, x_test_ind]
    pp = selector.pvalues_[x_test_ind]
    print("best 5 feature for 1k: " + ' '.join(str(e) for e in x_test_ind))
    print("best p values for 1k: " + ' '.join(str(e) for e in pp))

    clf.fit(X_new, y_1k)
    y_pred = clf.predict(x_test_new)
    confu = confusion_matrix(y_test, y_pred)
    accuracys.append(accuracy(confu))

    #=================== 32k 5 features train ========================
    selector = SelectKBest(f_classif, 5)
    X_new = selector.fit_transform(X_train, y_train)
    x_test_ind = selector.get_support(indices=True)
    x_test_new = X_test[:, x_test_ind]
    pp = selector.pvalues_[x_test_ind]
    print("best 5 feature for 32k: "+ ' '.join(str(e) for e in x_test_ind))
    print("best p values for 32k: "+ ' '.join(str(e) for e in pp))

    clf.fit(X_new, y_train)
    y_pred = clf.predict(x_test_new)
    confu = confusion_matrix(y_test, y_pred)
    accuracys.append(accuracy(confu))

    print(accuracys)
    #write accuracys append to file
    with open('a1_3.3.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(accuracys)

    return X_1k, y_1k

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    #print('TODO Section 3.4')
    npz = np.load(filename)
    feats = npz[npz.files[0]]
    X = feats[:, 0:173]
    y = feats[:, 173]

    output_csv = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        accuracys = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape, X_test.shape)

        # ============================ svc =====================================
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        confu = confusion_matrix(y_test, y_pred)

        accuracys.append(accuracy(confu))

        # ============================ svc gamma=2 =====================================
        #
        clf = SVC(gamma=2, max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        confu = confusion_matrix(y_test, y_pred)

        accuracys.append(accuracy(confu))

        # =============================random forest =================================

        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        confu = confusion_matrix(y_test, y_pred)

        accuracys.append(accuracy(confu))

        # ==================================== mlp =======================================
        clf = MLPClassifier(alpha=0.05)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        confu = confusion_matrix(y_test, y_pred)
        accuracys.append(accuracy(confu))

        # ========================= adaboost ===============================
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        confu = confusion_matrix(y_test, y_pred)

        accuracys.append(accuracy(confu))

        output_csv.append(accuracys)

    #write to csv
    with open('a1_3.4.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        for row in output_csv:
            spamwriter.writerow(row)


    s_array = []
    best_column = i-1
    acc_2d = np.array(output_csv)

    for column in range(acc_2d.shape[1]):
        if column != best_column:
            s = stats.ttest_rel(acc_2d[:,column],acc_2d[:,best_column])
            print(acc_2d[:,column])
            print(acc_2d[:,best_column])
            s_array.append(s.pvalue)

    with open('a1_3.4.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(s_array)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)

    X_1k, y_1k = class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)

    # TODO : complete each classification experiment, in sequence.
