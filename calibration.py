import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from TreeStructures import ClassificationTree
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from sklearn.calibration import CalibrationDisplay



def calibration_error(y_true, y_proba, n_bins = 5, strategy = 'uniform', normalize=False):
    if normalize:
        df_min_ = y_proba.min()
        df_max_ = y_proba.max()
        y_proba = (y_proba - df_min_) / (df_max_ - df_min_)

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy=strategy)
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
   



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str) 
    parser.add_argument('seed', type=int)
    parser.add_argument('-m','--models', help='<Required> models in pkl format delimited by -', required=True, type=str)
    args = parser.parse_args()


    seed = args.seed
    
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


   
    
    
    models = [ (pickle.load(open(model, 'rb')), model) for model in args.models.split(' ')]
    models[0][0].decisor = True

    strategy = 'quantile'
    f, ax = plt.subplots(1,1)
    plt.title('Calibration plots (reliability curve) \n'+args.dataset.split('/')[-1].split('.')[0])
    for (clf, name)  in models:
        if clf.decisor:
            #Standardization
            clf.print_tree_structure()
            scaler = StandardScaler()
            X= scaler.fit_transform(X)
            X_test = scaler.transform(X_test)
            y_prob = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test)
            print(name, balanced_accuracy_score(2*y_test-1, y_pred))
        
        else:
            #Normalization
            scaler = MinMaxScaler()
            X_train = scaler.fit(X)
            X_test_2  = scaler.transform(X_test)
            y_prob = clf.predict_proba(X_test_2)
            y_pred = clf.predict(X_test_2)
            print(name, balanced_accuracy_score(y_test, y_pred))
        
        
        disp = CalibrationDisplay.from_predictions(y_test, y_prob, ax=ax, n_bins=10, strategy=strategy)

        #print(y_test)
        print(y_prob)
    names = ['IDEAL MODEL', 'T-OLCT']
    ax.legend(names)
    f.savefig('calibration/calib_{}_{}.pdf'.format(args.dataset.split('/')[-1].split('.')[0], strategy))




    
    



