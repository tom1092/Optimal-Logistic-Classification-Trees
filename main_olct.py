import numpy as np
from gurobipy import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from TreeStructures import ClassificationTree
from itertools import product
import argparse
import csv
import pickle
from dt_greedy_growing import GreedyDecisionTree
from sklearn.metrics import balanced_accuracy_score, accuracy_score




class OLCTModel(BaseEstimator):

    def __init__(self, alpha_0: float =1, alpha_1: float  = 1, max_depth : int = 2, time_limit : int = 100, n_jobs: int = -1, v : str = 'v0', refine: str = 'standard'):

        """
        Initialize OLCT model instance with given parameters.
        
        :param: alpha_0: float, default=1
            Regularization parameter for weights of branch layers at depth 0.
        
        :param: alpha_1: float, default=1
            Regularization parameter for weights of branch layers at depth 1.
        
        :param: max_depth: int, default=2
            Maximum depth allowed for the decision stump tree.
        
        :param: time_limit: int, default=100
            Time limit in seconds for the optimization of model with Gurobi.
        
        :param: n_jobs: int, default=-1
            Number of CPU cores to be used for parallelizing the optimization with Gurobi.

        :param: refine: str, default='standard'
            Type of refinement used for the last branching level at the end of training:
                - 'standard' for standard refinement using logistic regression.
                - 'weighted' for weighted refinement using logistic regression with class_weight.
                - 'none' for no refinement.
        
        :param: v: str, default='v0'
            Which set of points to use for building the linear underapprossimation of log_loss
            from:
            Sato, Toshiki, et al. 
            "Feature subset selection for logistic regression via mixed integer optimization." 
            Computational Optimization and Applications 64 (2016): 865-880.
            
        """

        self.mio_tree = None
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.mean_n_weights = 0
        self.n_jobs = n_jobs
        self.v = v
        self.refine = refine
    
    
    def logistic_loss(self, v : np.array) -> np.array:
        """
        Compute the logistic loss of the given array.
        
        :param: v: np.array
            Input array to compute logistic loss on.
        
        :return: np.array
            Computed logistic loss of the given array.
        """

        return np.log(1+np.exp(-np.array(v)))

    def logistic_loss_derivative(self, v : np.array) -> np.array:

        """
        Compute the derivative of the logistic loss function for the given array.
        
        :param v: np.array
            Input array to compute derivative of logistic loss function on.
        
        :return: np.array
            Computed derivative of logistic loss function for the given array.
        """

        return -1/(1+np.exp(np.array(v)))


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
        min = np.inf

        eps = 1e-05
        M = 1e02
        
        #Mantaining the point 1e02 is equal to build an horizontal underestimator at level 0
        #This point can be removed as the constraint is equally imposed by the non negativity of slacks
        v = [-1e02,  0]
        if self.v == 'v1':
            v.extend([-1.90, 1.90])
        if self.v == 'v2':
            v.extend([-0.89, 0.89, -3.55, 3.55])

        f_v = self.logistic_loss(v)
        f_i_v = self.logistic_loss_derivative(v)
        
        K = [i for i in range(len(v))]

        if self.max_depth == 2:

            #Tree structure of depth = 2. Set of branches {0,1,2}. Set of leaves {3,4,5,6}
            T_b = [0, 1, 4]
            T_b_l = [1, 4]


            #Set of branches that makes the classification
            #T_b_l = [1, 4]

            
            
            C_r = {
                0: [4],
                1: [1],
                4: [4]
            }

            C_l = {
                0: [1],
                1: [1],
                4: [4]
            }

        

        elif self.max_depth==3:

            #Tree structure of depth = 3.
            #Set of branches
            T_b = [0, 1, 2, 5, 8, 9, 12]

            #Set of last layer of branches
            T_b_l = [2, 5, 9, 12]


            
            
            C_r = {
                0: [9, 12],
                1: [5],
                2: [2],
                5: [5],
                8: [12],
                9: [9],
                12: [12]
            }

            C_l = {
                0: [2, 5],
                1: [2],
                2: [2],
                5: [5],
                8: [9],
                9: [9],
                12: [12]
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
        




        #Z_i_t is 1 if the point i arrive at branch t in T_b_l
        z = self.model.addVars(list(range(len(X))), T_b_l, vtype = GRB.BINARY, lb = 0, ub = 1, name = "Z")

        #Branch biases
        b = self.model.addVars(T_b, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "b")

        #Branch weights 
        w = self.model.addVars(list(range(len(X[0]))), T_b, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "W")

        #Branch weights for 1 norm
        w_1_0= self.model.addVars(list(range(len(X[0]))), T_b, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "W10")

        w_1_1= self.model.addVars(list(range(len(X[0]))), T_b, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "W11")

        #Branch slacks
        e = self.model.addVars(list(range(len(X))), T_b, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "e")

        #Branch slacks linear approx
        e_hat = self.model.addVars(list(range(len(X))), T_b, vtype=GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "e_hat")

        
        #Constraints for logistic classification
        for i in range(len(X)):
            for t in T_b:
                for k in K:
                    self.model.addConstr(e_hat[i, t] >= f_v[k] + f_i_v[k]*(y[i]*(quicksum(w[j, t]*X[i, j] for j in range(len(X[0]))) + b[t]) - v[k]))
        

        #Constraints for slacks
        for i in range(len(X)):
            for t in T_b:
                self.model.addConstr(e[i, t] >= e_hat[i, t] - M*(1 - quicksum(z[i, l] for l in list(set(C_l[t]).union(set(C_r[t]))))))


        #Constraints for l1 weights
        for j in range(len(X[0])):
            for t in T_b:
                self.model.addConstr(w[j, t] == w_1_0[j, t] - w_1_1[j, t])
                
        

        #Constraints for tracking points
        for i in range(len(X)):
            for t in list(set(T_b) - set(T_b_l)):
                self.model.addConstr(quicksum(w[j, t]*X[i, j] for j in range(len(X[0]))) + b[t] >= eps - M*(1-quicksum(z[i, l] for l in C_r[t])))
                self.model.addConstr(quicksum(w[j, t]*X[i, j] for j in range(len(X[0]))) + b[t] <= M*(1-quicksum(z[i, l] for l in C_l[t])))

        

        #Each point has to arrive on exactly one leaf
        for point_index in range(len(X)):
            self.model.addConstr(quicksum(z[point_index, t] for t in T_b_l) == 1)


        


        if self.max_depth == 2:
            f = quicksum(w_1_0[j, t] + w_1_1[j, t] for t in T_b for j in range(len(X[0]))) + self.alpha_0*quicksum(e[i, 0] for i in range(len(X)))+self.alpha_1*quicksum(e[i, t] for i in range(len(X)) for t in [1, 4])

        if self.max_depth == 3:
            f = quicksum(w_1_0[j, t] + w_1_1[j, t] for t in T_b for j in range(len(X[0]))) + self.alpha_0*quicksum(e[i, 0] for i in range(len(X)))+self.alpha_1*quicksum(e[i, t] for i in range(len(X)) for t in [1, 2, 5, 8, 9, 12])


       
        

        self.model.setObjective(f)



        if warm_start:

            olct_tree = GreedyDecisionTree(min_samples_leaf = 1, max_depth = self.max_depth, split_strategy='logistic', C=1)
            olct_tree.fit(X, y)


            W_warm, b_warm = get_warm_start_from_tree(olct_tree.tree, X, y)

            for feature, branch_index in product(range(len(X[0])), range(len(b_warm))):
                w[feature, T_b[branch_index]].Start = W_warm[feature, branch_index]

            
            for branch_index in range(len(b_warm)):
                b[T_b[branch_index]].Start = b_warm[branch_index]
               



        self.model.setParam("IntFeasTol", 1e-09)
        self.model.setParam("TimeLimit", self.time_limit)
        self.model.setParam('OutputFlag', 1)
        self.model.setParam('Threads', self.n_jobs)
        self.model.setParam('MIPGap', 1e-8)
        self.model.optimize()

        if self.model.ObjVal < min:
            min = self.model.ObjVal
        if debug:
            for v in self.model.getVars():
                print("Var: {}, Value: {}".format(v.varName, v.x))


        #Create the solution tree 
        mio_tree = ClassificationTree(depth = self.max_depth, oblique=True, decisor=True)
        mio_tree.random_complete_initialize(len(X[0]))


        
        for branch_id in T_b:

            mio_tree.tree[branch_id].intercept = self.model.getVarByName(f'b[{branch_id}]').x
            weights = []
            for j in range(len(X[0])):
                weights.append(self.model.getVarByName(f'W[{j},{branch_id}]').x)

            if branch_id == 0:
                mio_tree.tree[branch_id].C = self.alpha_0
            else:
                mio_tree.tree[branch_id].C = self.alpha_1

            mio_tree.tree[branch_id].weights = np.array(weights)
            mio_tree.tree[branch_id].non_zero_weights_number = np.sum(np.abs(mio_tree.tree[branch_id].weights) > 1e-05)
           

        mio_tree.build_idxs_of_subtree(X, range(len(X)), mio_tree.tree[0], oblique=True)

        branches, leaves = ClassificationTree.get_leaves_and_branches(mio_tree.tree[0])
        
        self.mean_n_weights = np.mean([b.non_zero_weights_number for b in branches])
        
        #Refinement
        if self.refine == 'standard':
            mio_tree.refine_last_branch_layer(X, y, parallel=False)
        elif self.refine == 'weighted':
            mio_tree.refine_last_branch_layer(X, y, parallel=False, weighted=True)
        
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
        return 1 - ClassificationTree.misclassification_loss(self.mio_tree.tree[0], X, y, range(len(X)), decisor = True, oblique=True)

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

        return self.mio_tree.predict(X)

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
        
        
        param_dist = {'alpha_0': [1e-02,  1e-01, 1, 1e01, 1e02], 'alpha_1': [1e-02, 1e-01, 1, 1e01, 1e02]}
        
       

        #Cross Validation with 4 fold
        random_search = GridSearchCV(self, cv = 4, param_grid=param_dist, n_jobs=4, error_score='raise', scoring = 'balanced_accuracy', refit = False)

        random_search.fit(X, y)
        #best_estimator = random_search.best_estimator_
        best_params = random_search.best_params_

        #return best_estimator, best_params

        return best_params




def get_warm_start_from_tree(tree: ClassificationTree, X: np.array, y: np.array) -> tuple:

    """

    This function takes a ClassificationTree object and training data X and y as input, 
    and returns the weights w and intercepts b of the tree's branches as a tuple to initialize the warm-start procedure

    Parameters:

        :param: tree : ClassificationTree object - A trained decision tree classifier.
        :param: X : numpy array - Input features of shape (n_samples, n_features).
        :param: y : numpy array - Target variable of shape (n_samples,)

    
    Returns:

        :return: w : numpy array - Weights of the branches in the tree. Shape is (n_features, n_branches)

        :return: b : list - List of intercepts for each branch in the tree.
    """

    #Get leaves and branch sets
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
    parser.add_argument('--v', dest='v', type=str, default='0', help="Choose the set for log_loss approximation: 0, 1, 2")
    parser.add_argument('--transform', dest='transform', type=str, default='standard', help="Transformation for the data: standard, minmax")
    parser.add_argument('--refine', dest='refine', type=str, default='standard', help="Type of refinement: standard, none, weighted")
    args = parser.parse_args()


    olct_trains = []
    olct_tests = []
    olct_gaps = []
    olct_n_weights = []
    olct_runtimes = []
    olct_true_losses = []
    olct_mip_losses = []
    olct_gaps_true_loss = []

    #for seed in [0, 42, 314, 6, 71]:
    #for seed in [0, 314, 71]:
    for seed in [0]:

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


        #Scaling
        if args.transform == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        X  = scaler.fit_transform(X)
        X_test  = scaler.transform(X_test)

        

        mio_model = OLCTModel(max_depth = args.depth, time_limit = args.time, n_jobs = args.nt, v='v'+args.v, refine=args.refine)

        validation = args.validate

        if validation:
            #best_mio, best_params = mio_model.validate(X, 2*y - 1)
           
            mio_model.time_limit = 300 
            best_params = mio_model.validate(X, 2*y - 1)
            alphas = [best_params['alpha_0'], best_params['alpha_1']]
            mio_model.alpha_0 = alphas[0]
            mio_model.alpha_1 = alphas[1]

            mio_model.time_limit = args.time
            mio_model.fit(X, 2*y - 1, debug = args.debug)
            best_mio = mio_model



        else:
            best_mio = mio_model.fit(X, 2*y - 1, debug = args.debug)
            best_mio = mio_model


        olct_gaps.append(best_mio.model.MIPGap)
        olct_runtimes.append(best_mio.model.Runtime)
        olct_n_weights.append(best_mio.mean_n_weights)
        
        log_loss, reg_loss = best_mio.mio_tree.compute_log_loss(X, 2*y -1)

        true_loss = log_loss + reg_loss
        #true_loss = log_loss
        mip_loss = best_mio.model.objVal
        gap_true_loss = (true_loss - mip_loss) *100 / true_loss

        mio_tree = best_mio.mio_tree
        train_acc = 100*balanced_accuracy_score(2*y - 1, best_mio.predict(X))
        test_acc = 100*balanced_accuracy_score(2*y_test- 1, best_mio.predict(X_test))

        
        
        print("Log loss true: ", log_loss)
        print("Regularization loss true: ", reg_loss)
        print("Total True loss: ", log_loss + reg_loss)
        print("Gurobi loss: ", best_mio.model.objVal)

        olct_trains.append(train_acc)
        olct_tests.append(test_acc)
        olct_true_losses.append(true_loss)
        olct_mip_losses.append(mip_loss)
        olct_gaps_true_loss.append(gap_true_loss)



        result_line = []
        dataset = args.dataset.split('/')[-1]
        shape = str(X_data.shape[0])+" $\times$ "+str(X_data.shape[1])
        result_line.append(dataset)
        result_line.append(str(seed))
        result_line.append(str(round(train_acc, 2)))
        result_line.append(str(round(test_acc, 2)))
        result_line.append(str(round(best_mio.model.MIPGap * 1e02, 2)))
        result_line.append(str(round(best_mio.model.Runtime, 2)))
        result_line.append(str(round(true_loss, 2)))
        result_line.append(str(round(best_mio.model.objVal, 2)))
        result_line.append(str(round(gap_true_loss, 2)))
        result_line.append(str(round(best_mio.mean_n_weights, 2)))

        to_csv(args.out_file, result_line)


        print("\n"*3)
        print("**************OLCT****************")
        if validation:
            print("Best Params: ", best_params)
        print("Train acc on tree structure after gurobi: ", train_acc)
        print("Test acc on tree structure after gurobi: ", test_acc)
        mio_tree.print_tree_structure()
        print("\n"*3)

       
        #Save the model
        pickle.dump(mio_tree, open('model_Vs/olct_'+str(dataset)+'_v'+str(args.v)+'_'+'s'+str(seed)+'.pkl', 'wb'))



    print("OLCT Train: ", np.mean(olct_trains), " +- ", np.std(olct_trains))
    print("OLCT Test: ", np.mean(olct_tests), " +- ", np.std(olct_tests))
    print("OLCT gaps: ", np.mean(olct_gaps), " +- ", np.std(olct_gaps))
    print("OLCT times: ", np.mean(olct_runtimes), " +- ", np.std(olct_runtimes))
    print("OLCT mean number of weights per branch node: ", np.mean(olct_n_weights), " +- ", np.std(olct_n_weights))

    result_line = []

    dataset = args.dataset.split('/')[-1]
    shape = str(X_data.shape[0])+" $\times$ "+str(X_data.shape[1])
    result_line.append(dataset)

    result_line.append(str(round(np.mean(olct_trains), 2)) + " $\pm$ "+str(round(np.std(olct_trains), 2)))
    result_line.append(str(round(np.mean(olct_tests), 2)) + " $\pm$ "+str(round(np.std(olct_tests), 2)))
    result_line.append(str(round(np.mean(olct_gaps), 2)) + " $\pm$ "+str(round(np.std(olct_gaps), 2)))
    result_line.append(str(round(np.mean(olct_runtimes), 2)) + " $\pm$ "+str(round(np.std(olct_runtimes), 2)))
    result_line.append(str(round(np.mean(olct_true_losses), 2)) + " $\pm$ "+str(round(np.std(olct_true_losses), 2)))
    result_line.append(str(round(np.mean(olct_mip_losses), 2)) + " $\pm$ "+str(round(np.std(olct_mip_losses), 2)))
    result_line.append(str(round(np.mean(olct_gaps_true_loss), 2)) + " $\pm$ "+str(round(np.std(olct_gaps_true_loss), 2)))
    result_line.append(str(round(np.mean(olct_n_weights), 2)) + " $\pm$ "+str(round(np.std(olct_n_weights), 2)))


    to_csv(args.out_file, result_line)





    



        
