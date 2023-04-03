import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.svm import SVC
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from TreeStructures import ClassificationTree



def calibration_error(y_true, y_proba, n_bins = 5, strategy = 'uniform', normalize=False):
    if normalize:
        df_min_ = y_proba.min()
        df_max_ = y_proba.max()
        y_proba = (y_proba - df_min_) / (df_max_ - df_min_)

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy=strategy)
    # if len(prob_true) < n_bins:
    #     exit('Empty bin')
    weights = []
    for r in range(n_bins):
        w = np.count_nonzero([1 if r/n_bins<=y_proba[i] <= (r+1)/n_bins else 0 for i in range(len(y_proba))])/len(y_proba)
        weights.append(w)

    summ = 0
    j = 0
    for i in range(len(weights)):
        if weights[i] > 0:
            summ += weights[i]*np.abs(prob_pred[j] - prob_true[j])
            j += 1
    return summ 
    # return np.sum([weights[i]*np.abs(prob_pred[i] - prob_true[i]) for i in range(len(weights))])

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()


   
    
    
    for seed in [9]:

        np.random.seed(seed)

        print("\n"*5)
        print("DATASET: ", args.dataset)
        print("\n"*2)
        print("SEED: ", seed)


        #Dataset
        data = np.load(args.dataset)
        X_data = data[:, 1:]
        y_data = data[:, 0].astype(np.int64)
        print("Data Shape: ", X_data.shape)


        #80/20 split
        X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data)


        #Normalization
        scaler = StandardScaler()
        X  = scaler.fit_transform(X)
        X_test  = scaler.transform(X_test)
        

        
        clf = LogisticRegression(random_state=0, C = 0.1, penalty='l1', solver='liblinear').fit(X, y)
        y_preds = clf.predict_proba(X_test)[:, 1]
        
        print(calibration_error(y_test, y_preds))

        clf = LogisticRegression(random_state=0, C=0.1, class_weight='balanced', penalty='l1', solver='liblinear').fit(X, y)
        y_preds = clf.predict_proba(X_test)[:, 1]
        print(calibration_error(y_test, y_preds))

        #prob_true_b, prob_pred_b = calibration_curve(y_test, y_preds, n_bins=n_bins, strategy=strat)

        #print(prob_true_b)
        #print(prob_pred_b)


        svm = SVC(kernel = 'linear')
        svm.fit(X, y)
        y_preds_svm = svm.decision_function(X_test)

        # print(y_preds_svm)
       
        # print(calibration_curve(y_test, y_preds_svm, n_bins=5, normalize=True))

        print(calibration_error(y_test, y_preds_svm, normalize=True))
      

        dtree = DecisionTreeClassifier(max_depth=6, random_state=0)
        dtree.fit(X, y)
        tree = ClassificationTree()
        tree.initialize_from_CART(X, dtree)
        #tree.print_tree_structure()
        y_preds_tree = tree.predict_prob(points=X_test)
        
        print(calibration_error(y_test, y_preds_tree))