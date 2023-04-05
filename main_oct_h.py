import numpy as np
from gurobipy import *
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator
from scipy.stats import loguniform
from sklearn.datasets import load_svmlight_file
from TreeStructures import ClassificationTree
from itertools import product
import argparse
import csv
import pickle
from dt_greedy_growing import GreedyDecisionTree
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from calibration import calibration_error



class OCTModel(BaseEstimator):

    def __init__(self, alpha: float = 1, max_depth : int = 2, time_limit : int = 100, n_jobs: int = -1):

        """
        Initialize OCT model  (Bertsimas 2017) for oblique trees with given parameters.

        From original paper: 

        Bertsimas, Dimitris, and Jack Dunn. "Optimal classification trees." Machine Learning 106 (2017): 1039-1082.
        
        :param: alpha: float, default=1
            Penalty term for defining a branch node.
        
        :param: max_depth: int, default=2
            Maximum depth allowed for the decision stump tree.
        
        :param: time_limit: int, default=100
            Time limit in seconds for the optimization of model with Gurobi.
        
        :param: n_jobs: int, default=-1
            Number of CPU cores to be used for parallelizing the optimization with Gurobi.
        
    
        """

        self.mio_tree = None
        self.alpha = alpha
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.mean_n_weights = 0
        self.n_jobs = n_jobs
    
    
   
    def fit(self, X: np.array =None, y: np.array =None, debug : bool = False):
        

        """
        Fits the model to the given training data using the MIP binary logistic regression tree with regularization.

        :param X: A numpy array of shape (n_samples, n_features) containing the input features of the training data.
        :type X: np.array

        :param y: A numpy array of shape (n_samples,) containing the target labels of the training data.
        :type y: np.array

        :param debug: Whether to print debug messages during fitting. Defaults to False.
        :type debug: bool

        :return: None

        """
        
        self.model = Model()
        warm_start = True


        N = len(y)
        min_obj = np.inf

        eps = 1e-05
        M = 1e02
        



        if self.max_depth == 2:

            #Tree structure of depth = 2. Set of branches {0,1,2}. Set of leaves {3,4,5,6}
            T = [0, 1, 2, 3, 4, 5, 6]
            T_b = [0, 1, 4]
            T_l = [2, 3, 5, 6]


            #Set of branches that makes the classification
            T_b_l = [1, 4]

            
            
            A_r = {
                0: [],
                1: [],
                2: [],
                3: [1],
                4: [0],
                5: [0],
                6: [0, 4]
            }

            A_l = {
                0: [],
                1: [0],
                2: [1, 0],
                3: [0],
                4: [],
                5: [4],
                6: []
            }

            Anc = {
                0: [],
                1: [0],
                2: [0, 1],
                3: [0, 1],
                4: [0],
                5: [0, 4],
                6: [0, 4]
            }
        

        elif self.max_depth==3:

           #Tree structure of depth = 3. 
            T = list(range(15))
            T_b = [0,1,8,2,5,9,12]
            T_l = [3,4,6,7,10,11,13,14]


            #Set of branches that makes the classification
            T_b_l = [2,5,9,12]

            A_r = {
                0: [],
                1: [],
                2: [],
                3: [],
                4: [2],
                5: [1],
                6: [1],
                7: [1, 5],
                8: [0],
                9: [0],
                10: [0],
                11: [0, 9],
                12: [0, 8],
                13: [0, 8],
                14: [0, 8, 12]
                

            }

            A_l = {
                0: [],
                1: [0],
                2: [0, 1],
                3: [0, 1, 2],
                4: [0, 1],
                5: [0],
                6: [0, 5],
                7: [0],
                8: [],
                9: [8],
                10: [8, 9],
                11: [8],
                12: [],
                13: [12],
                14: []
            }

            Anc = {
                0: [],
                1: [0],
                2: [0, 1],
                3: [0, 1, 2],
                4: [0, 1, 2],
                5: [0, 1],
                6: [0, 1, 5],
                7: [0, 1, 5],
                8: [0],
                9: [0, 8],
                10: [0, 8, 9],
                11: [0, 8, 9],
                12: [0, 8],
                13: [0, 8, 12],
                14: [0, 8, 12]
            }


        elif self.max_depth==1:
            
             #Tree structure of depth = 1. Set of branches {0,1,2}. Set of leaves {3,4,5,6}
            T = [0, 1, 2]
            T_b = [0]
            T_l = [1, 2]


            #Set of branches that makes the classification
            T_b_l = [0]

            A_r = {
                0: [],
                1: [],
                2: [0]
            }

            A_l = {
                0: [],
                1: [0],
                2: []
            }
            

            Anc = {
                0: [],
                1: [0],
                2: [0],
            }
        


        #Baseline loss: misclass obtained simply by predicting the most popular class.
        #Thus the misclass is just the number of elements of the minority class

        L_hat = min(np.count_nonzero(y), N - np.count_nonzero(y))


        #N_min is set as 5% of total number of training samples as in the original paper
        N_min = int(0.05*len(y))

 
        # Y_i_k is 1 if x_i has class k. -1 o.w.        
        # Binary classification K = 2
        Y = np.zeros(shape=(len(X), 2))

        for i in range(len(X)):
            Y[i, :] = np.array([1 if y[i] == k else -1 for k in [0, 1]])


        #Z_i_t is 1 if the point i arrive at leaf t in T_l
        Z = self.model.addVars(list(range(len(X))), T_l, vtype = GRB.BINARY, lb = 0, ub = 1, name = "Z")

        #A_j_t are the coefficients of hyperplanes for each node
        A = self.model.addVars(list(range(len(X[0]))), T_b, vtype = GRB.CONTINUOUS, lb = -1, ub = 1, name = "A")

        #A_j_t_tilde are the coefficients of hyperplanes for each node in abs value
        A_tilde = self.model.addVars(list(range(len(X[0]))), T_b, vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "A_tilde")

        #Track if the leaf t has any point
        l = self.model.addVars(T_l, vtype = GRB.BINARY, lb = 0, ub = 1, name = "l")

        #Branch thresholds are box constraints
        b = self.model.addVars(T_b, vtype = GRB.CONTINUOUS, lb = -1, ub = 1, name = "b")

        #Number of point of class k for each leaf (binary classification)
        N = self.model.addVars([0,1], T_l, vtype = GRB.CONTINUOUS, name="N")

        #Total number of points in each leaf
        N_t = self.model.addVars(T_l, vtype = GRB.CONTINUOUS, name="N_t")

        #Binary variables to decide wheter to split on branch t
        d = self.model.addVars(T_b, vtype = GRB.BINARY, name = 'd', lb = 0, ub = 1)

        #Binary variables to decide wheter to split on branch t
        S = self.model.addVars(list(range(len(X[0]))), T_b, vtype = GRB.BINARY, name = 's', lb = 0, ub = 1)


        #c_k_t track the prediction. 1 if leaf t has label k else 0.
        C = self.model.addVars([0, 1], T_l, vtype = GRB.BINARY, name="C")


        #Misclassification loss for each leaf
        L = self.model.addVars(T_l, vtype=GRB.CONTINUOUS, name = "L", lb=0)


        #Constraints for the loss on each leaf
        for leaf_index in T_l:
            for k in [0, 1]:
                self.model.addConstr(L[leaf_index] >= N_t[leaf_index] - N[k, leaf_index]-len(y)*(1-C[k, leaf_index]))
                self.model.addConstr(L[leaf_index] <= N_t[leaf_index] - N[k, leaf_index]+len(y)*C[k, leaf_index])


        #Total number of k class points in leaf t
        for leaf_index in T_l:
            for k in [0,1]:
                self.model.addConstr(0.5 * quicksum(Z[i, leaf_index]*(1+Y[i, k]) for i in range(len(Y))) == N[k, leaf_index])


        #Total number of point for each leaf constraint
        for leaf_index in T_l:
            self.model.addConstr(quicksum(Z[i, leaf_index] for i in range(len(X))) == N_t[leaf_index])

        
        #Set c_k_t to make a single class prediction
        for leaf_index in T_l:
            self.model.addConstr(quicksum(C[k, leaf_index] for k in [0, 1]) == l[leaf_index])

        
        #Constraints for tracking points
        for i in range(len(X)):
            for t in T_l:
                if A_r[t] != []:
                    for m in A_r[t]:
                        self.model.addConstr(quicksum(A[j, m]*X[i, j] for j in range(len(X[0]))) >= b[m] + eps -M*(1-Z[i, t]))
                if A_l[t] != []:
                    for m in A_l[t]:
                        self.model.addConstr(quicksum(A[j, m]*X[i, j] for j in range(len(X[0]))) <=  b[m]+M*(1-Z[i, t]))

        

        #Each point has to arrive on exactly one leaf
        for point_index in range(len(X)):
            self.model.addConstr(quicksum(Z[point_index, t] for t in T_l) == 1)


        #For each leaf t, z_i_t has to be less than l_t
        for leaf_index in T_l:
            for i in range(len(X)):
                self.model.addConstr(Z[i, leaf_index] <= l[leaf_index])

        

        #Min point number on each leaf
        for leaf_index in T_l:
            self.model.addConstr(quicksum(Z[i, leaf_index] for i in range(len(X))) >= N_min*l[leaf_index])


        #For each branch, the node has to split on maximum d features
        for branch_index in T_b:
            self.model.addConstr(quicksum(A_tilde[f, branch_index] for f in range(len(X[0]))) <= d[branch_index])


        #Get absolute values
        for branch_index in T_b:
            for j in range(len(X[0])):
                self.model.addConstr(A_tilde[j, branch_index] >= A[j, branch_index])
                self.model.addConstr(A_tilde[j, branch_index] >= -A[j, branch_index])
                self.model.addConstr(A[j, branch_index] >= -S[j, branch_index])
                self.model.addConstr(A[j, branch_index] <= S[j, branch_index])
                self.model.addConstr(S[j, branch_index] <= d[branch_index])




        
        for branch_index in T_b:
            self.model.addConstr(quicksum(S[j, branch_index] for j in range(len(X[0]))) >= d[branch_index])


        #Tree structure: If the node t splits, then also all the ancestors have to split   
        for branch_index in T_b:
            for anc_index in Anc[branch_index]:
                self.model.addConstr(d[branch_index] <= d[anc_index])
        
        
        
        #The threshold b_t has to be box constr in [-d_t, d_t]
        for branch_index in T_b:
            self.model.addConstr(b[branch_index] <= d[branch_index])
            self.model.addConstr(b[branch_index] >= -d[branch_index])
        


        f = 1/L_hat*quicksum(L[t] for t in T_l) + self.alpha*quicksum(S[j, t] for j in range(len(X[0])) for t in T_b)
        #f = quicksum(L[t] for t in T_l)

        self.model.setObjective(f)





        if warm_start:

            greedy_tree = GreedyDecisionTree(min_samples_leaf = N_min, max_depth = self.max_depth, split_strategy='oct')
           
            greedy_tree.fit(X, y)


            A_warm, b_warm,  = get_warm_start_from_tree(greedy_tree.tree, X, y)
            
            for feature, branch_index in product(range(len(X[0])), range(len(b_warm))):
                A[feature, T_b[branch_index]].Start = A_warm[feature, branch_index]

            
            for branch_index in range(len(b_warm)):
                b[T_b[branch_index]].Start = -b_warm[branch_index]
               



        self.model.setParam("IntFeasTol", 1e-09)
        self.model.setParam("TimeLimit", self.time_limit)
        self.model.setParam('OutputFlag', 1)
        self.model.setParam('Threads', self.n_jobs)
        self.model.setParam('MIPGap', 1e-8)
        self.model.optimize()

        if self.model.ObjVal < min_obj:
            min_obj= self.model.ObjVal
        if debug:
            for v in self.model.getVars():
                print("Var: {}, Value: {}".format(v.varName, v.x))


        #Create the solution tree 
        mio_tree = ClassificationTree(depth = self.max_depth, oblique=True)
        mio_tree.random_complete_initialize(len(X[0]))

        for branch_id in T_b:
            mio_tree.tree[branch_id].intercept = -self.model.getVarByName(f'b[{branch_id}]').x
            mio_tree.tree[branch_id].weights = np.zeros(len(X[0]))
            for feat in range(len(X[0])):
                val = self.model.getVarByName(f'A[{feat},{branch_id}]').x
                mio_tree.tree[branch_id].weights[feat] = val
            mio_tree.tree[branch_id].non_zero_weights_number = np.sum(np.abs(mio_tree.tree[branch_id].weights) > 1e-05)

        
        mio_tree.build_idxs_of_subtree(X, range(len(X)), mio_tree.tree[0], oblique=True)

        #Set labels on the leaves
        for leaf_id in T_l:
            node = mio_tree.tree[leaf_id]      
            if node.data_idxs:
                best_class = np.bincount(y[node.data_idxs]).argmax()
                node.value = best_class
                node.pos_prob = np.count_nonzero(y[node.data_idxs])/len(node.data_idxs)

        self.mio_tree = mio_tree
        
    

        mio_tree.build_idxs_of_subtree(X, range(len(X)), mio_tree.tree[0], oblique=True)

        branches, leaves = ClassificationTree.get_leaves_and_branches(mio_tree.tree[0])
        
        self.mean_n_weights = np.mean([b.non_zero_weights_number for b in branches])
        
        self.mio_tree = mio_tree

        


    def score(self, X: np.array, y: np.array) -> float:
        
        """
        Returns the score of the trained ClassificationTree model on the input data.

        Parameters:

        :param: X : np.array
            The input feature matrix of shape (n_samples, n_features).
        :param: y : np.array
            The true class labels for the input data of shape (n_samples,).

        Returns:

        score : float
            The score of the trained model on the input data, computed as 1 minus the misclassification loss.

        """

        return 1 - ClassificationTree.misclassification_loss(self.mio_tree.tree[0], X, y, range(len(X)), decisor = False, oblique=True)

    def predict(self, X: np.array) -> np.array:

        """
        Returns the predicted class labels for the input data using the trained ClassificationTree model.

        Parameters:
       
        :param: X : np.array
            The input feature matrix of shape (n_samples, n_features).

        Returns:
        
        :param: y_pred : np.array
            The predicted class labels for the input data of shape (n_samples,).

        """

        return ClassificationTree.predict_label(X, self.mio_tree.tree[0], oblique = True, decisor = False)

    def validate(self, X: np.array, y: np.array) -> tuple:

        """
        Performs cross-validation with 4 folds on the input data and returns the best hyperparameters and the corresponding
        trained model.

        Parameters:
        
        :param: X : np.array
            The input feature matrix of shape (n_samples, n_features).
        :param: y : np.array
            The true class labels for the input data of shape (n_samples,).

        Returns:
        
        :param: best_estimator : ClassificationTree
            The best trained model based on the cross-validation.
        :param: best_params : dict
            The best hyperparameters found during the cross-validation.

        """
        
        param_dist = {'alpha': [2**(i) for i in range(-8, 3)].append(0)}
        
       

        #Cross Validation with 4 fold
        random_search = GridSearchCV(self, cv = 4, param_grid=param_dist, n_jobs=4, error_score='raise', scoring = 'balanced_accuracy')

        random_search.fit(X, y)
        best_estimator = random_search.best_estimator_
        best_params = random_search.best_params_

        return best_estimator, best_params




def get_warm_start_from_tree(tree: ClassificationTree, X: np.array, y: np.array) -> tuple:

    """

    This function takes a ClassificationTree object and training data X and y as input, 
    and returns the weights w and intercepts b of the tree's branches as a tuple to initialize the warm-start procedure

    Parameters:

        :param: tree : ClassificationTree object - A trained decision tree classifier.
        :param: X : numpy array - Input features of shape (n_samples, n_features).
        :param: y : numpy array - Target variable of shape (n_samples,)

    
    """

    #Get leaves set
    branches, leaves = ClassificationTree.get_leaves_and_branches(tree.tree[0])
    leaves.sort(key = lambda x: x.id)
    branches.sort(key = lambda x: x.id)
    
    
    #SET W_j_t
    w = np.zeros(shape = (len(X[0]), len(branches)))
    for j in range(len(X[0])):
        w[j, :] = [branches[i].weights[j] for i in range(len(branches))]

    #SET b_t
    b = [branch.intercept for branch in branches]

    
    return w, b


def to_csv(filename : str , row : list):

    """
    
    This function takes a filename and a list of data rows as input, and appends the rows to a CSV file.

    Parameters:
    filename : str - Name of the CSV file.
    row : list - List of data to be written to the CSV file.

    """

    with open(filename, 'a') as f:
        writer = csv.writer(f, delimiter=',',)
        writer.writerow(row)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true', help="See variables from gurobi")
    parser.add_argument('--time', dest='time', type=int, default=30, help="Time for gurobi solver")
    parser.add_argument('--nt', dest='nt', type=int, default=8, help="Number of threads")
    parser.add_argument('--validate', dest='validate', action='store_true', help="Validate the model")
    parser.add_argument('--out', dest = 'out_file', type=str, default='log.txt')
    parser.add_argument('--depth', dest='depth', type=int, default=2, help="Max depth for the tree")
    args = parser.parse_args()


    oct_trains = []
    oct_tests = []
    oct_gaps = []
    oct_n_weights = []
    oct_runtimes = []
    
    for seed in [0, 42, 314, 6, 71]:
    #for seed in [0]:

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
        scaler = MinMaxScaler()
        X  = scaler.fit_transform(X)
        X_test  = scaler.transform(X_test)

        

        mio_model = OCTModel(max_depth = args.depth, time_limit = args.time, n_jobs = args.nt)

        validation = args.validate

        if validation:
            best_mio, best_params = mio_model.validate(X, y )
        else:
            best_mio = mio_model.fit(X, y, debug = args.debug)
            best_mio = mio_model


        oct_gaps.append(best_mio.model.MIPGap)
        oct_runtimes.append(best_mio.model.Runtime)
        oct_n_weights.append(best_mio.mean_n_weights)


        mio_tree = best_mio.mio_tree
        train_acc = 100*balanced_accuracy_score(y, best_mio.predict(X))
        test_acc = 100*balanced_accuracy_score(y_test, best_mio.predict(X_test))

        #y_preds_tree = mio_tree.predict_prob(points=X_test)
        
        #print(calibration_error(y_test, y_preds_tree))

        print("Gurobi loss: ", best_mio.model.objVal)

        oct_trains.append(train_acc)
        oct_tests.append(test_acc)
        

        result_line = []
        dataset = args.dataset.split('/')[-1]
        shape = str(X_data.shape[0])+" $\times$ "+str(X_data.shape[1])
        result_line.append(dataset)
        result_line.append(str(seed))

        result_line.append(str(round(test_acc, 2)))
        result_line.append(str(round(best_mio.model.MIPGap * 1e02, 2)))
        result_line.append(str(round(best_mio.model.Runtime, 2)))

        result_line.append(str(round(best_mio.mean_n_weights, 2)))
    

        to_csv(args.out_file, result_line)


        print("\n"*3)
        print("**************OCT****************")
        if validation:
            print("Best Params: ", best_params)
        print("Train acc on tree structure after gurobi: ", train_acc)
        print("Test acc on tree structure after gurobi: ", test_acc)
        mio_tree.print_tree_structure()
        print("\n"*3)

        if seed==0:
            #Save the model
            pickle.dump(mio_tree, open('oct_h_'+str(dataset)+'.pkl', 'wb'))



    print("OCT Train: ", np.mean(oct_trains), " +- ", np.std(oct_trains))
    print("OCT Test: ", np.mean(oct_tests), " +- ", np.std(oct_tests))
    print("OCT gaps: ", np.mean(oct_gaps), " +- ", np.std(oct_gaps))
    print("OCT times: ", np.mean(oct_runtimes), " +- ", np.std(oct_runtimes))
    print("OCT mean number of weights per branch node: ", np.mean(oct_n_weights), " +- ", np.std(oct_n_weights))

    result_line = []

    dataset = args.dataset.split('/')[-1]
    shape = str(X_data.shape[0])+" $\times$ "+str(X_data.shape[1])
    result_line.append(dataset)

    result_line.append(str(round(np.mean(oct_trains), 2)) + " $\pm$ "+str(round(np.std(oct_trains), 2)))
    result_line.append(str(round(np.mean(oct_tests), 2)) + " $\pm$ "+str(round(np.std(oct_tests), 2)))
    result_line.append(str(round(np.mean(oct_gaps), 2)) + " $\pm$ "+str(round(np.std(oct_gaps), 2)))
    result_line.append(str(round(np.mean(oct_runtimes), 2)) + " $\pm$ "+str(round(np.std(oct_runtimes), 2)))
    result_line.append(str(round(np.mean(oct_n_weights), 2)) + " $\pm$ "+str(round(np.std(oct_n_weights), 2)))
   


    to_csv(args.out_file, result_line)





    



        
