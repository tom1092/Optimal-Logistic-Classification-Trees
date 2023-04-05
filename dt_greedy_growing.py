import numpy as np
from sklearn.datasets import make_blobs
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from TreeStructures import TreeNode, ClassificationTree
from sklearn.base import BaseEstimator
import sys
from tqdm.auto import tqdm
from time import sleep
import argparse
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_svmlight_file
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform, uniform
from sklearn.tree import plot_tree
from sklearn.metrics import f1_score, accuracy_score
from gurobipy import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

class GreedyDecisionTree(BaseEstimator):

    def __init__(self, C: float = 1.0, min_samples_leaf: int = 1, min_impurity: float = 1e-2,
                 max_depth: int = 4, split_strategy: str = 'scalar_svm') -> None:
        
        """
        Initializes a greedy classification tree model instance.

        Parameters:
            :param: C (float): Penalty parameter for slack in case of SVM trees or Logistic trees. Defaults to 1.0.
            :param: min_samples_leaf (int): Minimum number of samples required to be at a leaf node. Defaults to 1.
            :param: min_impurity (float): Threshold for early stopping in tree growth. Defaults to 0.01.
            :param: max_depth (int): Maximum depth of the tree. Defaults to 4.
            :param: split_strategy (str): The type of splitting criterion to be used for the decision tree.
                                  Defaults to 'scalar_svm'.

        """

        self.min_samples_leaf = min_samples_leaf
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.tree = None
        self.C = C
        self.split_strategy= split_strategy


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        """
        Fits the model to the training data.

        Parameters:

            :param: X (numpy.ndarray): Training input samples of shape (n_samples, n_features).
            :param: y (numpy.ndarray): Target values of shape (n_samples,).
        """

        oblique = False

        if self.split_strategy in ['svm', 'logistic', 'oct']:
            oblique = True

        root = self.train_tree(X, y)

        self.tree = ClassificationTree(oblique = oblique)

        self.tree.initialize(X, root)

        ClassificationTree.restore_tree(self.tree, X, y)



    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:

        """
        Calculates the accuracy score of the model.

        Parameters:
            :param: X (numpy.ndarray): Test input samples of shape (n_samples, n_features).
            :param: y_true (numpy.ndarray): True target values of shape (n_samples,).

        Returns:

            :return: float: The accuracy score of the model.
        """


        preds = self.predict(X)
        return self.tree.score(preds, y_true)

    def get_n_leaves(self) -> int:

        """
        Returns the number of leaves in the fitted tree.

        Returns:
            int: Number of leaves in the fitted tree.
        """
        return self.tree.n_leaves


    def get_depth(self)-> int:

        """
        Returns the depth of the fitted tree.

        Returns:
            int: Depth of the fitted tree.
        """

        return self.tree.get_depth(self.tree.tree[0])

    def print_tree(self):

        """
        Prints the structure of the fitted tree.

        """
        self.tree.print_tree_structure()

    def predict(self, X):

        """
        Predicts the target values for the given input samples.

        Parameters:
            :param: X (numpy.ndarray): Input samples of shape (n_samples, n_features).

        Returns:
            :return: numpy.ndarray: Predicted target values of shape (n_samples,).

        """

        return self.tree.predict_data(X, self.tree.tree[0])


    def svm_valid_scorer(self, svm_model: LinearSVC, X: np.ndarray, y: np.ndarray) -> float:

        """
        Calculate the loss function for a given linear SVM model on the validation set.

        N.B. It returns the negative loss, for maximization during validation

        Parameters:
            :param: svm_model (LinearSVC): The trained linear SVM model.
            :param: X (np.ndarray): The feature matrix of the validation set.
            :param: y (np.ndarray): The target vector of the validation set.

        Returns:
            float: The negative loss value of the linear SVM model on the validation set.

        """
        b = svm_model.intercept_[0]
        w = svm_model.coef_[0, 0]
        loss = 0.5*(w**2)+self.C*np.sum([max(0, 1 - y[j]*(w*X[j, i] + b)) for j in range(len(X))])
        return -loss



    def get_best_split_gini(self, X: np.ndarray, y: np.ndarray, idxs: np.ndarray) -> tuple:

        """
        Finds the best split according to the Gini index criterion.

        Parameters:
    
        :param: X : numpy.ndarray
            The input features of the dataset.
        :param: y : numpy.ndarray
            The target variable of the dataset.
        :param: idxs : numpy.ndarray
            The indices of the subset of the dataset on which to perform the search.

        Returns:
        :return tuple[int, float]
            The best feature index and the best threshold value to perform the split.

        """

        min_impur = np.inf
        best_f = None
        best_th = None
        for feature in range(len(X[0])):
            indices = np.argsort(X[idxs, feature])
            subset_X = X[idxs[indices], feature]
            subset_y = y[idxs[indices]]

            i = 0
            n_left, n_right = 0, len(indices)
            n_positive_left = 0
            n_negative_left = 0
            n_positive_right = np.count_nonzero(subset_y==1)
            n_negative_right = n_right - n_positive_right

            while (i < len(subset_X) - 2):

                th = (subset_X[i]+subset_X[i+1])/2

                if subset_y[i] == -1:
                    n_negative_left += 1
                    n_negative_right -= 1
                else:
                    n_positive_left += 1
                    n_positive_right -= 1


                k = 1
                while k+i < len(subset_X)-1 and subset_X[k+i] == subset_X[i]:
                    if subset_y[k+i] == -1:
                        n_negative_left += 1
                        n_negative_right -= 1
                    else:
                        n_positive_left += 1
                        n_positive_right -= 1
                    k+=1

                i+=k
                n_left += k
                n_right += -k

                gini_impurity_left = n_positive_left/n_left * (1 - n_positive_left/n_left) + n_negative_left/n_left * (1 - n_negative_left/n_left)
                gini_impurity_right = n_positive_right/n_right * (1 - n_positive_right/n_right) + n_negative_right/n_right * (1 - n_negative_right/n_right)

                weighted_gini = n_left/len(indices) * gini_impurity_left + n_right/len(indices) * gini_impurity_right
                if weighted_gini < min_impur:
                    min_impur = weighted_gini
                    best_f = feature
                    best_th = th

        return best_f, best_th


    def get_best_split_svm_scalar(self, X_data: np.ndarray, y_data: np.ndarray, idxs: list) -> tuple:

        """
        Finds the best feature and split threshold using the best scalar SVM among the feature.

        Parameters:
            :param: X_data (np.ndarray): The feature matrix.
            :param: y_data (np.ndarray): The target vector.
            :param: idxs (list): The indices of the data points to consider.

        Returns:
            tuple[int, float, float]: The index of the best feature, the intercept of the SVM model, and the weight of the best feature.

        """
        X = X_data[idxs]
        y = y_data[idxs]
        N = len(y)
        best_model = None
        best_loss = np.inf
        best_feat = 0
        for j in range(len(X[0])):

            clf = LinearSVC(loss = 'hinge', max_iter = 1e06)
            clf.fit(X[:, j].reshape((-1, 1)), y)
            summma_slacks = 0
            for i in range(len(X)):
                summma_slacks += max(0, 1-(2*y[i] - 1)*(clf.coef_[0, 0] * X[i, j] + clf.intercept_))

            loss = summma_slacks + 0.5 * clf.coef_[0, 0]
            if loss < best_loss:
                best_loss = loss
                best_model = clf
                best_feat = j


        
        return best_feat,  best_model.intercept_[0], best_model.coef_[0, 0]
        


    def get_best_split_svm(self, X: np.ndarray, y: np.ndarray, idxs: np.ndarray) -> tuple:

        """

        Returns the intercept and coefficients for the linear SVM classifier trained on the given subset of data.
        
        Parameters:

            :param: X (numpy.ndarray): The input features of shape (n_samples, n_features).
            :param: y (numpy.ndarray): The target values of shape (n_samples,).
            :param: idxs (numpy.ndarray): The indices of the subset of data to use for training.
        
        Returns:
            :return: A tuple containing the intercept and coefficients of the SVM classifier.

        """

        X = X[idxs]
        y = y[idxs]
        clf = LinearSVC(C= self.C)
        clf.fit(X, y)
        return clf.intercept_, clf.coef_.reshape((len(X[0])),)


    def get_best_split_logistic(self, X: np.ndarray, y: np.ndarray, idxs: np.ndarray) -> tuple:

        """
        Returns the intercept and coefficients for the logistic regression classifier trained on the given subset of data.
        
        Parameters:
            :param: X (numpy.ndarray): The input features of shape (n_samples, n_features).
            :param: y (numpy.ndarray): The target values of shape (n_samples,).
            :param: idxs (numpy.ndarray): The indices of the subset of data to use for training.
        
        Returns:
            :return: A tuple containing the intercept and coefficients of the logistic regression classifier.

        """

        X = X[idxs]
        y = y[idxs]
        clf = LogisticRegression(C= self.C, penalty='l1', solver='liblinear')
        clf.fit(X, y)
        return clf.intercept_, clf.coef_.reshape((len(X[0])),)


    def get_best_split_oct(self, X: np.ndarray, y: np.ndarray, idxs: np.ndarray, Nmin:int, time:int) -> tuple:

        """
        Returns the intercept and coefficients for the single node OCT loss trained on the given subset of data.
        
        Parameters:
            :param: X (numpy.ndarray): The input features of shape (n_samples, n_features).
            :param: y (numpy.ndarray): The target values of shape (n_samples,).
            :param: idxs (numpy.ndarray): The indices of the subset of data to use for training.
            :param: Nmin (int): Minimum number of sample for each leaf.
            :param: time (int): Time Limit for gurobi solver
        
        Returns:
            :return: A tuple containing the intercept and coefficients of the classifier .

        """

        X = X[idxs]
        y = y[idxs]
        N = len(y)

        model = Model()


        #In this warm start setting we have to solve a subproblem of just 1 branch node and 2 leaves
        #Thus the set of branches and leaves are just the following:
        T_l = [1, 2]
        T_b = [0]
        eps = 1e-05
        M = 1e02


        # Binary classification K = 2
        Y = np.zeros(shape=(len(X), 2))

        for i in range(len(X)):
            Y[i, :] = np.array([1 if y[i] == k else -1 for k in [0, 1]])


        #A_j are the coefficients of hyperplanes for the node
        A = model.addVars(list(range(len(X[0]))), vtype = GRB.CONTINUOUS, lb = -1, ub = 1, name = "A")

        #A_tilde are the coefficients of hyperplanes for the node in abs
        A_tilde = model.addVars(list(range(len(X[0]))), vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "A_tilde")

        #b is the intercept of the hyperplane
        b = model.addVar(vtype = GRB.CONTINUOUS, lb = -1, ub = 1, name = "b")

        #Z_i_t is 1 if the point i arrive at leaf t in T_l
        Z = model.addVars(list(range(len(X))), T_l, vtype = GRB.BINARY, lb = 0, ub = 1, name = "Z")

        #Number of point of class k for each leaf (binary classification)
        N = model.addVars([0,1], T_l, vtype = GRB.CONTINUOUS, name="N")

        #Total number of points in each leaf
        N_t = model.addVars(T_l, vtype = GRB.CONTINUOUS, name="N_t")

        #c_k_t track the prediction. 1 if leaf t has label k else 0.
        C = model.addVars([0, 1], T_l, vtype = GRB.BINARY, name="C")

        #Misclassification loss for each leaf
        L = model.addVars(T_l, vtype=GRB.CONTINUOUS, name = "L", lb=0)

        #Track if the leaf t has any point
        l = model.addVars(T_l, vtype = GRB.BINARY, lb = 0, ub = 1, name = "l")


        #Constraints for the loss on each leaf
        for leaf_index in T_l:
            for k in [0, 1]:
                model.addConstr(L[leaf_index] >= N_t[leaf_index] - N[k, leaf_index]-len(y)*(1-C[k, leaf_index]))
                model.addConstr(L[leaf_index] <= N_t[leaf_index] - N[k, leaf_index]+len(y)*C[k, leaf_index])


        #Total number of k class points in leaf t
        for leaf_index in T_l:
            for k in [0,1]:
                model.addConstr(0.5 * quicksum(Z[i, leaf_index]*(1+Y[i, k]) for i in range(len(Y))) == N[k, leaf_index])


        #Total number of point for each leaf constraint
        for leaf_index in T_l:
            model.addConstr(quicksum(Z[i, leaf_index] for i in range(len(X))) == N_t[leaf_index])

        
        #Set c_k_t to make a single class prediction
        for leaf_index in T_l:
            model.addConstr(quicksum(C[k, leaf_index] for k in [0, 1]) == l[leaf_index])

        
        #Constraints for tracking points
        for i in range(len(X)):
            model.addConstr(quicksum(A[j]*X[i, j] for j in range(len(X[0]))) + b >= eps-M*(Z[i, 1]))
            model.addConstr(quicksum(A[j]*X[i, j] for j in range(len(X[0]))) + b <= M*(1-Z[i, 1]))       

        

        #Each point has to arrive on exactly one leaf
        for point_index in range(len(X)):
            model.addConstr(quicksum(Z[point_index, t] for t in T_l) == 1)



        #Min point number on each leaf
        for leaf_index in T_l:
            model.addConstr(quicksum(Z[i, leaf_index] for i in range(len(X))) >= Nmin*l[leaf_index])



        #Absolute value
        for j in range(len(X[0])):
                model.addConstr(A_tilde[j] >= A[j])
                model.addConstr(A_tilde[j] >= -A[j])

        
        #1 norm equal to 1
        model.addConstr(quicksum(A_tilde[f] for f in range(len(X[0]))) == 1)


        #For each leaf t, z_i_t has to be less than l_t
        for leaf_index in T_l:
            for i in range(len(X)):
                model.addConstr(Z[i, leaf_index] <= l[leaf_index])
        
        
        f =  quicksum(L[t] for t in T_l)


        model.setObjective(f)
        model.setParam("IntFeasTol", 1e-09)

        #Time limit for warm start is 30s we set each sub problem to a time limit
        #which is equal to 30/n_branches thus 30/(2^D - 1) 
        model.setParam("TimeLimit", time)
        model.setParam('Threads', 1)
        model.setParam('MIPGap', 1e-8)
        #model.setParam('OutputFlag', 0)
        model.optimize()

        #Create the solution
        intercept = model.getVarByName(f'b').x
        weights = np.zeros(len(X[0]))
        for j in range(len(X[0])):
            weights[j] = model.getVarByName(f'A[{j}]').x


        return intercept, weights





    def train_tree(self, X: np.ndarray, y: np.ndarray) -> TreeNode:

        """
        Train a decision tree using the provided training data and return the root node of the tree.

        Parameters:
            :param: X (np.ndarray): The training features.
            :param: y (np.ndarray): The training labels.

        Returns:
            :return: object (TreeNode): The root node of the fitted decision tree.

        """
        key = 0
        depth = 0

        node = TreeNode(key, depth)
        node.data_idxs = np.array(range(len(X)))
        node.depth = 0
        node.is_leaf = True
        positive = np.count_nonzero(y)
        negative = len(node.data_idxs) - positive
        stack = [node]
        while(stack):

            n = stack.pop()

            ones = np.count_nonzero(y[n.data_idxs]==1)
           
            nums = np.array([len(n.data_idxs) - ones, ones])
            if nums[0] > nums[1]:
                n.value = -1
            else:
                n.value = 1


            n.impurity = len(n.data_idxs) - max(nums[0], nums[1])
            if (n.depth == self.max_depth) or nums[0] <= 2 or nums[1] <= 2 or (len(n.data_idxs) == self.min_samples_leaf):
                n.is_leaf = True
            else:
                
                #Create axis-aligned split with gini
                if self.split_strategy == 'gini':

                #SET node attributes to the best
                    n.threshold = None
                    n.feature, n.threshold = self.get_best_split_gini(X, y, np.array(n.data_idxs))


                    #Get indexes of left and right subset
                    indexes_left = np.array([i for i in n.data_idxs if X[i, n.feature] <= n.threshold])
                    indexes_right = np.array(list(set(n.data_idxs) - set(indexes_left)))

                #Create oblique split using linear SVM
                elif self.split_strategy == 'svm':
                    n.intercept, n.weights = self.get_best_split_svm(X, y, np.array(n.data_idxs))

                    #Get indexes of left and right subset
                    indexes_left = np.array([i for i in n.data_idxs if np.dot(n.weights, X[i, :]) + n.intercept <= 0])
                    indexes_right = np.array(list(set(n.data_idxs) - set(indexes_left)))
                
                ##Create oblique split using logistic
                elif self.split_strategy == 'logistic':
                    n.intercept, n.weights = self.get_best_split_logistic(X, y, np.array(n.data_idxs))

                    #Get indexes of left and right subset
                    indexes_left = np.array([i for i in n.data_idxs if np.dot(n.weights, X[i, :]) + n.intercept <= 0])
                    indexes_right = np.array(list(set(n.data_idxs) - set(indexes_left)))

                ##Create oblique split using OCT method
                elif self.split_strategy == 'oct':

                    #For OCT the whole warm start phase has a duration of 30s.
                    #This time has to be divided for each branch node to get the time for each subproblem
                    #(Sequential setting)
                    oct_time = int(30/(2**self.max_depth - 1))
                    n.intercept, n.weights = self.get_best_split_oct(X, y, np.array(n.data_idxs), self.min_samples_leaf, oct_time)

                    #Get indexes of left and right subset
                    indexes_left = np.array([i for i in n.data_idxs if np.dot(n.weights, X[i, :]) + n.intercept <= 0])
                    indexes_right = np.array(list(set(n.data_idxs) - set(indexes_left)))


                #Create axis aligned split using scalar svm
                else:
        
                    n.threshold = None
                    n.feature, n.intercept, n.w = self.get_best_split_svm_scalar(X, y, np.array(n.data_idxs))

                    n.threshold = -n.intercept/n.w
                
                    #Get indexes of left and right subset
                    indexes_left = np.array([i for i in n.data_idxs if X[i, n.feature] <= n.threshold])
                    indexes_right = np.array(list(set(n.data_idxs) - set(indexes_left)))

                

                #If you have enough points on the children then you have to split
                if (len(indexes_left) >= self.min_samples_leaf and len(indexes_right) >= self.min_samples_leaf):
                    
                    
                    n.is_leaf = False
                    #Create two children
                    n_left = TreeNode(key+1, depth+1)
                    n_right = TreeNode(key+2, depth+1)
                    n_left.depth = n.depth + 1
                    n_right.depth = n.depth + 1
                    n_left.data_idxs = indexes_left
                    n_right.data_idxs = indexes_right
                    key = key + 2
                    #Set the father
                    n.left_node = n_left
                    n.right_node = n_right
                    n.left_node_id = n_left.id
                    n.right_node_id = n_right.id

                    ones = ones = np.count_nonzero(y[n_left.data_idxs]==1)
                    nums = np.array([len(n_left.data_idxs) - ones, ones])
                    if nums[0] > nums[1]:
                        n_left.value = -1
                    else:
                        n_left.value = 1
                    res_left = len(n_left.data_idxs)-np.abs(nums[0]-nums[1])


                    ones = ones = np.count_nonzero(y[n_right.data_idxs]==1)
                    nums = np.array([len(n_right.data_idxs) - ones, ones])
                    if nums[0] > nums[1]:
                        n_right.value = -1
                    else:
                        n_right.value = 1
                    res_right = len(n_right.data_idxs)-np.abs(nums[0]-nums[1])


                    n_left.is_leaf = False
                    n_right.is_leaf = False

                    if len(n_left.data_idxs) == 1 or len(n_right.data_idxs) == 1:
                        n_right.is_leaf = True
                        n_left.is_leaf = True
                
                    else:
                        n_left.is_leaf = True
                        n_right.is_leaf = True
                        stack.append(n_right)
                        stack.append(n_left)

                else:
                    ones = np.count_nonzero(y[n.data_idxs]==1)
                    nums = np.array([len(n.data_idxs) - ones, ones])
                    if nums[0] > nums[1]:
                        n.value = -1
                    else:
                        n.value = 1
                    n.is_leaf = True


        return node




