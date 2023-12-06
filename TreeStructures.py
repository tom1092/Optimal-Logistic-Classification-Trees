from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression


class TreeNode:

    def __init__(self, 
                 id: int, 
                 depth: int, 
                 left_node_id: int = None , 
                 right_node_id: int = None, 
                 feature: int = None, 
                 threshold: float = None, 
                 is_leaf: bool = None, 
                 value: int =  None):
        

        """
        This method is the constructor for a TreeNode object, which is a general node in a Classification Tree (CT).
        It initializes the object with the following parameters:

        Parameters:

            :param: id (int): The unique identifier of the node.
            :param: depth (int): The depth of the node in the tree.
            :param: left_node_id (int): The unique identifier of the left child node. Default is None.
            :param: right_node_id (int): The unique identifier of the right child node. Default is None.
            :param: feature (int): The index of the feature used for splitting the node for axis-aligned CT. Default is None.
            :param: threshold (float): The threshold value used for splitting the node. Default is None.
            :param: is_leaf (bool): Whether the node is a leaf node or not. Default is None.
            :param: value (int): The predicted value of the node. Default is None.

        Attributes:

            id (int): The unique identifier of the node.
            depth (int): The depth of the node in the tree.
            left_node_id (int): The unique identifier of the left child node. Initialized as None.
            right_node_id (int): The unique identifier of the right child node. Initialized as None.
            left_node (TreeNode): The left child node. Initialized as None.
            right_node (TreeNode): The right child node. Initialized as None.
            feature (int): The index of the feature used for splitting the node. Initialized as None.
            threshold (float): The threshold value used for splitting the node. Initialized as None.
            is_leaf (bool): Whether the node is a leaf node or not. Initialized as None.
            parent_id (int): The unique identifier of the parent node. Initialized as -1.
            value (int): The predicted value of the node. Initialized as None.
            data_idxs (list of int): The indices of the data points in the node. Initialized as an empty list.
            weights (numpy array): The weights of the data points in the node. Initialized as None.
            C (float): The regularization hyperparameter for oblique trees with norm penalty. Initialized as 1.
            w (numpy array): The weights for the linear SVM. Initialized as None.
            non_zero_weights_number (int): The number of non-zero weights for the linear SVM. Initialized as -1.
            intercept (float): The intercept for the linear SVM. Initialized as None.
            prob (float): The probability for the linear SVM. Initialized as None.
            impurity (float): The impurity of the node. Initialized as None.

        """
        self.id = id
        self.depth = depth
        self.left_node_id = left_node_id
        self.right_node_id = right_node_id
        self.left_node = None
        self.right_node = None
        self.feature = feature
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.parent_id = -1
        self.value = value
        self.data_idxs = []
        self.weights = None

        #Regularization hyper for oblique trees with norm penalty
        self.C = 1
        
        #For scalar svm
        self.w = None
        self.non_zero_weights_number = -1
        self.intercept = None
        self.pos_prob = None
        self.impurity = None

    @staticmethod
    def copy_node(node: TreeNode) -> TreeNode:
        """
            Given a TreeNode object it returns a new TreeNode with the same attributes.
            
            Parameters:
                :param: node (TreeNode): The node you want to copy.
                
            Returns:
                return: object (TreeNode): The new copy of the given object.
        """
        new = TreeNode(node.id, node.depth, node.left_node_id, node.right_node_id, node.feature, node.threshold, node.is_leaf, node.value)
        new.parent_id = node.parent_id
        new.data_idxs = node.data_idxs
        new.impurity = node.impurity
        new.weights = node.weights
        new.intercept = node.intercept
        new.w = node.w
        new.left_node = node.left_node
        new.right_node = node.right_node
        return new



class ClassificationTree:

    def __init__(self, min_samples:int  =None, 
                 oblique : bool = False, 
                 depth : int = None, 
                 decisor : bool = False):
        
        """
            This method is the constructor for a Tree object, 
            which represents a classification tree or a 'decisor' tree
            i.e. a structure where each branch is a linear classifier (svm/logistic)
            in these cases the final label is predicted by the last branch rather than by the leaf
            on which the point fall. 
            It initializes the object with the following parameters:

            Parameters:

                :param: min_samples (int): The minimum number of samples required to split a node. Default is None.
                :param: oblique (bool): Whether the tree is oblique or not. Default is False.
                :param: depth (int): The maximum depth of the tree. Default is None.
                :param: decisor (bool): Whether the tree is a 'decisor' tree or not. Default is False.

            Attributes:

                tree (dict): A dictionary representing the tree structure.
                min_samples (int): The minimum number of samples required to split a node.
                depth (int): The maximum depth of the tree.
                n_leaves (int): The number of leaf nodes in the tree. Initialized as 0.
                oblique (bool): Whether the tree is oblique or not.
                n_nodes (int): The number of nodes in the tree. Initialized as None.
                decisor (bool): Whether the tree is a 'decisor' tree or not
        """

        self.tree = {}
        self.min_samples = min_samples
        self.depth = depth
        self.n_leaves = 0
        self.oblique = oblique
        self.n_nodes = None

        
        self.decisor = decisor


    
    def initialize(self, 
                   data : np.array, 
                   root_node : TreeNode):
        

        """
        This method initializes a binary tree structure based on a dictionary data input. 
        The method takes two parameters:

        Parameters:

            :param data: The np.array containing training data to initialize the tree.
            :param root_node: The root node of the tree.

        The method creates the tree structure by performing a depth-first search traversal of the tree. 
        It initializes the root node, sets the depth of the tree, and adds it to the tree structure. 
        Then, it traverses the tree by iterating through the nodes on a stack.

        After initializing all the nodes, the method sets the parent ids for each node, and builds the sets of indexes for each node. 
        The sets of indexes represent the data samples that are covered by each node.

        Overall, this method initializes a binary tree structure based on a dictionary structure and builds the sets of indexes for each node.

        """
        self.depth = self.get_depth(root_node)

        
        stack = [root_node]
        while(stack):
            n = stack.pop()
            n = TreeNode.copy_node(n)
            self.tree[n.id] = n

            
            if not n.is_leaf:
                self.tree[n.id].left_node_id = n.left_node.id
                self.tree[n.id].right_node_id = n.right_node.id

                stack.append(n.right_node)
                stack.append(n.left_node)
            else:
                self.n_leaves += 1
                if n.value==-1:
                    n.value = 0

        #Set each father
        for i in range(len(self.tree)):
            #If it's a branch node
            if self.tree[i].left_node_id != self.tree[i].right_node_id:
                self.tree[self.tree[i].left_node_id].parent_id = i
                self.tree[self.tree[i].right_node_id].parent_id = i
                self.tree[self.tree[i].id].left_node = self.tree[self.tree[i].left_node_id]
                self.tree[self.tree[i].id].right_node = self.tree[self.tree[i].right_node_id]

        #Build the sets of indexes for each node
        self.build_idxs_of_subtree(data, range(len(data)), self.tree[0], oblique = self.oblique)




    def random_complete_initialize(self, n_features:int):


        """
        This method creates a complete (balanced) axis-aligned classification tree
        using a randomized intialization both for the feature and threshold of each node.
        Threshold are initialized with uniform in [0, 1].


        Parameters:
            :param: n_features: the number of features of the data
        """

        id = 0
        depth = 0
        root = TreeNode(id, depth)
        stack = [root]
        

        while stack:
            node = stack.pop()
            if node.depth<self.depth:
                left_node = TreeNode(id+1, node.depth + 1, is_leaf=False)
                right_node = TreeNode(id+2, node.depth + 1, is_leaf=False)
                node.left_node = left_node
                node.right_node = right_node
                node.feature = np.random.randint(low = 0, high = n_features)
                node.threshold = np.random.uniform()
                stack.append(right_node)
                stack.append(left_node)
                id = id + 2
            else:
                node.is_leaf = True

        id = -1

        #DFS to mantain coherence in ids
        stack = [root]
        while stack:
            node = stack.pop()     
            id += 1
            node.id = id
            self.tree[id] = node
            if not node.is_leaf:
                stack.append(node.right_node)
                stack.append(node.left_node)

        stack = [root]
        while stack:
            node = stack.pop()
            if node.left_node:  
                node.left_node_id = node.left_node.id
            if node.right_node:
                node.right_node_id = node.right_node.id
            if not node.is_leaf:
                stack.append(node.right_node)
                stack.append(node.left_node)




    def initialize_from_CART(self, data: np.array, clf: DecisionTreeClassifier):

        """
        This function initializes the classification tree object
        given a CART (Classification and Regression Tree) structure of the scikit-learn tree object. 

        Parameters:
            :param: data: the np.array containing the train data
            :param: clf: the fitted sklearn DecisionTreeClassifier 

        """

        self.depth = clf.tree_.max_depth
        n_nodes = clf.tree_.node_count
        self.n_nodes = n_nodes
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        value = clf.tree_.value

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
                self.tree[node_id] = TreeNode(node_id, node_depth[node_id], children_left[node_id], children_right[node_id], feature[node_id], threshold[node_id], False, -1)
                if self.oblique:
                    ej = np.zeros(len(data[0]))
                    ej[feature[node_id]] = 1
                    self.tree[node_id].weights = ej
                    self.tree[node_id].intercept = -threshold[node_id]
            else:
                is_leaves[node_id] = True
                self.tree[node_id] = TreeNode(node_id, node_depth[node_id], -1, -1, feature[node_id], threshold[node_id], True, np.argmax(value[node_id]))
                self.tree[node_id].pos_prob = value[node_id][0][1]/np.sum(value[node_id])
                #self.tree[node_id].pos_prob = value[node_id]
                self.n_leaves += 1

        #Set father
        for i in range(len(self.tree)):
            #Test if it is a branch
            if self.tree[i].left_node_id != self.tree[i].right_node_id:
                self.tree[self.tree[i].left_node_id].parent_id = i
                self.tree[self.tree[i].right_node_id].parent_id = i
                self.tree[self.tree[i].id].left_node = self.tree[self.tree[i].left_node_id]
                self.tree[self.tree[i].id].right_node = self.tree[self.tree[i].right_node_id]

        #Build the subsets of indexes given train data of each node
        self.build_idxs_of_subtree(data, range(len(data)), self.tree[0], oblique = self.oblique)
        self.depth = self.get_depth(self.tree[0])



    def get_depth(self, root:TreeNode):

        """
            This method compute a DFS to infer the depth of the sub-tree rooted at root

            Parameters:

                :param: root: The TreeNode object which is the root of the subtree of which you want to know the depth
        """

        stack = [root]
        depth = 0
        while(stack):
            actual = stack.pop()
            if actual.depth > depth:
                depth = actual.depth
            if not actual.is_leaf:
                stack.append(actual.left_node)
                stack.append(actual.right_node)
        return depth




    def predict_p(self, point: np.array, root: TreeNode):

        """
        This method return the predicted label for a single point, starting from the node rooted at root.

        Parameters:

            :param: point: the point you want to predict.
            :param: root: The TreeNode object which is the root of the classification tree

        """

        actual = root
        while(not actual.is_leaf):
            if point[actual.feature] < actual.threshold:
                actual = actual.left_node
            else:
                actual = actual.right_node
        return actual.value
    



    def predict_data(self, data: np.array, root: TreeNode):

        """
            Return the np array of predictions for each sample in data starting from the node 'root'.

            Parameters:

                :param: data: the np.array containing the points you want to predict

        """
        return np.array([self.predict_p(p, root) for p in data])


    
    @staticmethod
    def copy_tree(tree: ClassificationTree):

        """
            This method create a copy of the Classification Tree object given as parameter

            Parameters:

                :param: tree: The Classification Tree object that you want to get a copy of
        """

        new = ClassificationTree(min_samples=tree.min_samples, oblique = tree.oblique)
        new.depth = tree.depth
        new.n_leaves = tree.n_leaves
        for (node_id, node) in tree.tree.items():
            new.tree[node_id] = TreeNode(node_id, node.depth, node.left_node_id, node.right_node_id, None, None, node.feature, node.threshold, node.is_leaf, node.value)
            new.tree[node_id].parent_id = node.parent_id
            new.tree[node_id].data_idxs = node.data_idxs
            new.tree[node_id].weights = node.weights
            new.tree[node_id].intercept = node.intercept

        stack = [new.tree[0]]
        while stack:
            actual = stack.pop()
            if not actual.is_leaf:
                actual.left_node = new.tree[actual.left_node_id]
                actual.right_node = new.tree[actual.right_node_id]
                stack.append(actual.left_node)
                stack.append(actual.right_node)

        return new


    @staticmethod
    def get_nodes_at_depth(depth: int, tree: ClassificationTree):
        """
            This method returns the list of nodes at a the given depth of the Classification Tree object
            given as parameter

            Parameters:

                :param: depth: the depth of the nodes you want to retrieve
                :param: tree: the ClassificationTree object
        """
        nodes = []
        for (id, node) in tree.tree.items():
            if node.depth == depth:
                nodes.append(node)
        return nodes



    
    def print_tree_structure(self):

        """
            This method prints the structure of the tree in a readable way.

            Parameters:


        """

        print("The binary tree has %s nodes and has "
              "the following structure:"
              % len(self.tree))

        if self.oblique:

            for i in self.tree.keys():
               
                if self.tree[i].is_leaf:
                    print("%snode=%s is child of node %s. It's a leaf node. Np: %s - Imp: %s - Value: %s" % (self.tree[i].depth * "\t", i, self.tree[i].parent_id, len(self.tree[i].data_idxs), self.tree[i].impurity, self.tree[i].value))
                else:
                    if self.tree[i].non_zero_weights_number <= 1:

                        coef_idx = np.argmax(np.abs(self.tree[i].weights))
                        coef = self.tree[i].weights[coef_idx]
                        print("%snode=%s is child of node %s. It's a oblique test node node. C: %s Np: %s - Imp: %s - non-zero-weights: %s - feature index: %s - Next =  %s if %sx + %s <= 0 else "
                            "%s."
                            % (self.tree[i].depth * "\t",
                                i,
                                self.tree[i].parent_id,
                                self.tree[i].C,
                                len(self.tree[i].data_idxs),
                                self.tree[i].impurity,
                                self.tree[i].non_zero_weights_number,
                                coef_idx,
                                self.tree[i].left_node_id,
                                round(coef, 3),
                                round(float(self.tree[i].intercept), 3),
                                self.tree[i].right_node_id,
                                ))

                    else:
                        print("%snode=%s is child of node %s. It's a oblique test node node. C: %s Np: %s - Imp: %s - non-zero-weights: %s - Next =  %s if w^T x + %s <= 0 else "
                            "%s."
                            % (self.tree[i].depth * "\t",
                                i,
                                self.tree[i].parent_id,
                                self.tree[i].C,
                                len(self.tree[i].data_idxs),
                                self.tree[i].impurity,
                                self.tree[i].non_zero_weights_number,
                                self.tree[i].left_node_id,
                                round(float(self.tree[i].intercept), 3),
                                self.tree[i].right_node_id,
                                ))
        else:
            for i in self.tree.keys():
                if self.tree[i].is_leaf:
                    print("%snode=%s is child of node %s. It's a leaf node. Np: %s - Imp: %s - Value: %s - PosProb: %s" % (self.tree[i].depth * "\t", i, self.tree[i].parent_id, len(self.tree[i].data_idxs), self.tree[i].impurity, self.tree[i].value, self.tree[i].pos_prob))
                else:
                    print("%snode=%s is child of node %s. It's a test node. Np: %s - Imp: %s - Next =  %s if X[:, %s] <= %s else "
                        "%s."
                        % (self.tree[i].depth * "\t",
                            i,
                            self.tree[i].parent_id,
                            len(self.tree[i].data_idxs),
                            self.tree[i].impurity,
                            self.tree[i].left_node_id,
                            self.tree[i].feature,
                            self.tree[i].threshold,
                            self.tree[i].right_node_id,
                            ))


   
    def build_idxs_of_subtree(self, data: np.array, idxs: np.array, root_node: TreeNode, oblique: bool  = False):
        
        """
            This method is for building the sets of indexes of the data points 
            that reach each node in a classification tree rooted at node root_node

            Parameters:

                :param: data: Train data
                :param: idxs: the indexes of the data points to be used for building the tree
                :param: root_node: the root node of the decision tree
                :param: oblique: a boolean value that specifies whether the decision tree is oblique (i.e., has split planes that are not aligned with the feature axes)

        """

        #First empty each previous index set for each node
        stack = [root_node]
        while(stack):
            actual_node = stack.pop()
            actual_node.data_idxs = []
            if actual_node.left_node and actual_node.right_node:
                actual_node.left_node.data_idxs = []
                actual_node.right_node.data_idxs = []
                stack.append(actual_node.left_node)
                stack.append(actual_node.right_node)

        #See the path of the point in the tree to infer which node the point reaches
        for i in idxs:
            path = ClassificationTree.get_path_to(data[i], root_node, oblique)
            for node in path:
                node.data_idxs.append(i)



    
    @staticmethod
    def get_path_to(x: np.array, root_node: TreeNode, oblique: bool):

        """
            Starting from the node 'root_node', the method returns the list of the nodes which
            the point x follows to arrive at a leaf.

            Parameters:

                :param: x: The point of which you want to know the path in the tree.
                :param: root_node: The root node of the tree.
                :param: oblique: Whether the tree is axis-aligned or oblique
        """

        #Start from the node 
        actual_node = root_node
        path = [actual_node]

        #Check if the CT is oblique 
        if oblique:
            #Keep going till a leaf it's reached
            while(not actual_node.is_leaf):
                weights = actual_node.weights
                intercept = actual_node.intercept

                #1e-09 is to avoid numerical issues with MIP solvers
                if np.dot(x, weights) + intercept >= 1e-09:
                    actual_node = actual_node.right_node
                else:
                    actual_node = actual_node.left_node
                path.append(actual_node)
        
        #Here we have parallel split CT
        else:
            #Keep going till a leaf it's reached
            while(not actual_node.is_leaf):
                #Decido quale sarà il prossimo figlio
                feature = actual_node.feature
                thresh = actual_node.threshold
                if x[feature] - thresh >= 1e-09:
                    actual_node = actual_node.right_node
                else:
                    actual_node = actual_node.left_node
                path.append(actual_node)

        return path


    
    def compute_log_loss(self, X:np.array, y:np.array, regularizer:int = 1):
        
        """
            Compute the loss of the logistic tree as the summation of the log_loss
            and the regularization penalty.

            Parameters:

                :param: X: The train set
                :param: y: The array of the labels (-1, 1)
                :param: regularizer: An int that specify which norm has to be taken into account for the regularization term

        """

        log_loss = 0
        reg_loss = 0

        #Compute the log loss for each point
        for i in range(len(X)):
            actual_node = self.tree[0]
            pred = actual_node.value
            while(not actual_node.is_leaf):
                weights = actual_node.weights
                intercept = actual_node.intercept
                log_loss += actual_node.C * np.log(1+np.exp(-y[i]*(np.dot(X[i], weights) + intercept)))

                if np.dot(X[i], weights) + intercept <= 0:                   
                    actual_node = actual_node.left_node
                else:                 
                    actual_node = actual_node.right_node

        #Compute the regularization loss for each branch
        #We go for a DFS
        stack = [self.tree[0]]
        while(stack):
            actual_node = stack.pop()
            weights = actual_node.weights
            reg_loss += np.linalg.norm(weights, regularizer)
            if not actual_node.left_node.is_leaf:
                stack.append(actual_node.left_node)
            if not actual_node.right_node.is_leaf:
                stack.append(actual_node.right_node)

        return log_loss, reg_loss
    


    
    def refine_last_branch_layer(self, X: np.array, y:np.array, parallel: bool = False, metric: str = 'loss', weighted: bool = False, penalty: str = 'l1'):
        

        """
            Perform a refinement of the last branching layer for logistic classification trees.
            Using the logistic loss.

            Parameters:

                :param: X: Train data
                :param: y: Label data
                :param: parallel: Whether the tree is axis-aligned or not
                :param: metric: The metric to use to choose the best feature for axis-aligned split (loss/bacc)
                :param: weighted: Whether to use class weights or not to fit the logistic regression model


        """

        #Search for each last branch
        stack = [self.tree[0]]
        last_branches = []
        while (stack):
            actual = stack.pop()
            if (actual.left_node.is_leaf):
                last_branches.append(actual)
            else:
                stack.append(actual.left_node)
                stack.append(actual.right_node)

        for branch in last_branches:

            #If the branch contains points and it's not pure
            if (len(X[branch.data_idxs]) > 0 and len(set(y[branch.data_idxs])) > 1):

                if weighted:
                    weighting_strategy = 'balanced'
                else:
                    weighting_strategy = None

                if not parallel:
                    if penalty:
                        lr = LogisticRegression(class_weight = weighting_strategy, penalty = penalty, solver = 'saga', C = branch.C).fit(X[branch.data_idxs], y[branch.data_idxs])

                    else:
                        lr = LogisticRegression(class_weight = weighting_strategy, solver = 'saga', penalty='none').fit(X[branch.data_idxs], y[branch.data_idxs])

                    
                    branch.weights = np.squeeze(lr.coef_)
                    branch.intercept = float(lr.intercept_)

                else:
                    #Get the best logistic regression model on a single feature
                    best_weights = None
                    best_loss = np.inf

                    #For each feature
                    for j in range(len(X[0])):
                        lr = LogisticRegression(class_weight = weighting_strategy, penalty = penalty, solver = 'saga', C = branch.C).fit(X[branch.data_idxs, j].reshape((-1, 1)), y[branch.data_idxs])
                        weights = np.zeros(len(X[0]))
                        weights[j] = lr.coef_[0]
                        loss = 0

                        #Best feature/threshold is the one with min log_loss
                        if metric == 'loss':
                            #Compute the log loss
                            for i in range(len(branch.data_idxs)):
                                loss += branch.C * np.log(1+np.exp(-y[branch.data_idxs[i]]*(np.dot(X[branch.data_idxs[i]], weights) + lr.intercept_[0])))

                            #Sum the l1 regularization
                            if penalty == 'l1':
                                loss += np.linalg.norm(weights, 1)
                        
                        #Best feature/threshold is the one with min balanced accuracy error
                        elif metric == 'bacc':
                            y_preds = lr.predict(X[branch.data_idxs, j].reshape((-1, 1)))
                            loss = 1 - balanced_accuracy_score(y_preds, y[branch.data_idxs])

                        if loss < best_loss:
                            best_loss = loss
                            best_weights = weights
                            best_intercept = lr.intercept_[0]

                    branch.weights = best_weights
                    branch.non_zero_weights_number = np.sum(np.abs(branch.weights) > 1e-05)
                    branch.intercept = float(best_intercept)

        self.build_idxs_of_subtree(X, range(len(y)), self.tree[0], oblique=True)


    
    @staticmethod
    def get_label_decisor_trees(x: np.array, root_node: TreeNode):
        """
            Get the predicted label for the point x in the case of 'decisor' classification trees
            i.e. in these structures the prediction is made by the last branch node rather than the leaf

            Parameters:

                :param: x: the point you want to get the label.
                :param: root_node: The root node of the structure


            Returns:
                :return: The predicted label for the point x
                :return: The associated score

        """
        actual_node = root_node
        pred = actual_node.value
        while(not actual_node.is_leaf):
            weights = actual_node.weights
            intercept = actual_node.intercept
            score = np.dot(x, weights) + intercept
            if score <= 0:
                pred = -1
                actual_node = actual_node.left_node
            else:
                pred = 1
                actual_node = actual_node.right_node

        return pred, score



    
    
    def predict(self, data: np.array):

        """
            Get the label predicted by the tree structure rooted at root_node
            for each point in data.

            Paramaters:

                :param: data: The data you want to predict.
            
            Returns:

                :return: The predicted labels for each point in data

        """

        if self.decisor:
            predictions = [self.get_label_decisor_trees(x, self.tree[0])[0] for x in data[:,]]
        else:

            predictions = [self.get_path_to(x, self.tree[0], self.oblique)[-1].value for x in data[:,]]
            predictions = [predictions[i] if predictions[i]!=None else 0 for i in range(len(predictions))]
        return predictions


    
    def predict_proba(self, data: np.array, type = 'logistic'):

        """
            Get the prob to be predicted as positive for the given points

            Parameters:

                :param: data: The points to be predicted shape (n_samples, n_features)
                :param: type: How to compute the prob. 'logistic' for logistic regression, 'leaf' for the leaf prob, svm to normalize the score with the svm loss

            Returns:
                
                :return: The prob to be predicted as positive for the given points
                
        """

        if self.decisor:
            if type == 'logistic':
                probas = [1/(1+np.exp(-self.get_label_decisor_trees(x, self.tree[0])[1])) for x in data[:,]]
                probas = [float(probas[i]) for i in range(len(probas))]
            
            elif type == 'svm':
                scores = [self.get_label_decisor_trees(x, self.tree[0])[1] for x in data[:,]]
                minimum = np.min(scores)
                maximum = np.max(scores)
                probas = (scores-minimum)/(maximum-minimum)

        else:
            probas = [self.tree[self.predict_leaf(point, self.tree[0], self.oblique)].pos_prob for point in data]
        return probas

    
    
    @staticmethod
    def predict_leaf(x: np.array, root_node: TreeNode, oblique: bool):

        """
            Returns the id of the leaf on which the point x fall.

            Parameters:

                :param: x: The point you want to predict.
                :param: root_node: The TreeNode object that is the root of the structure.
                :param: oblique: Wheter the tree performs oblique splits. 
        """
        path = ClassificationTree.get_path_to(x, root_node, oblique)
        return path[-1].id



   
    @staticmethod
    def misclassification_loss(root_node: TreeNode, 
                               data: np.array, 
                               target: np.array, 
                               indexes: np.array, 
                               oblique: bool =False, 
                               decisor: bool = False):

        """
            Get the misclassification loss of the subtree rooted at root_node on data respect to the targets.py

            Parameters:

                :param: root_node: The TreeNode object that is the root of the tree.
                :param: data: Data of which you want to compute the misclassification loss.
                :param: target: Labels
                :param: indexes: Is the array of indexes of data of the points you want to compute the misclassification loss
                :param: oblique: Wheter the tree performs oblique splits
                :param: decisor: Wheter the tree is a decisior tree

        """

        if len(indexes) > 0:
            preds = ClassificationTree.predict_label(data[indexes], root_node, oblique, decisor)
            n_misclassified = np.count_nonzero(target[indexes]-preds)
            return n_misclassified/len(indexes)
        else:
            return 0



    
    @staticmethod
    def restore_tree(tree: ClassificationTree, X: np.array, y: np.array):

        """
            Restore the dictionary structure of the tree using a DFS


            Parameters:

                :param: tree: The ClassificationTree object that has to be restored.
                :param: X: Train data.
                :param: y: labels.


        """
        T = tree.tree
        root_node = T[0]
        T.clear()
        stack = [root_node]
        depth = 0
        id = 0
        leaves = []
        while(len(stack) > 0):
            actual_node = stack.pop()
            actual_node.id = id
            ones = np.count_nonzero(y[actual_node.data_idxs])
            zeros = len(actual_node.data_idxs) - ones
            impurity = len(actual_node.data_idxs) - max(ones, zeros)
            actual_node.impurity = impurity          
            if not actual_node.is_leaf:
                actual_node.right_node.parent_id = actual_node.id
                actual_node.left_node.parent_id = actual_node.id
                stack.append(actual_node.right_node)
                stack.append(actual_node.left_node)              
            else:
                leaves.append(actual_node.id)
            id += 1
        tree.n_leaves = len(leaves)
        
        #Rearrange the dictionary to make the print more understandable
        stack = [root_node]
        while stack:
            actual_node = stack.pop()
            T[actual_node.id] = actual_node
            if not actual_node.is_leaf:
                stack.append(actual_node.left_node)   
                stack.append(actual_node.right_node)
        
    

    
    @staticmethod
    def get_leaves_and_branches(root: TreeNode):

        """
            Get leaves and branch nodes of the tree rooted at root

            Parameters:

                :param: root: TreeNode object which is the root of the tree

        """

        leaves = []
        branches = []
        stack = [root]
        while (stack):          
            actual = stack.pop()           
            if actual.is_leaf:
                leaves.append(actual)
            else:
                branches.append(actual)
                stack.append(actual.left_node)
                stack.append(actual.right_node)
        
        return branches, leaves



    
    def compute_prob(self, X: np.array, labels: np.array):

        """
            Compute the positive probabilities for each leaf

            Parameters:

                :param: X: Train data.
                :param: labels: Array of the labels
        """

        root = self.tree[0]
        stack = [root]
        while stack:
            actual = stack.pop()
            if actual.is_leaf:
                leaf_labels = labels[actual.data_idxs]
                n = len(leaf_labels)
                n_positive = np.count_nonzero(leaf_labels == 1)
                actual.prob = n_positive/n
            else:
                stack.append(actual.left_node)
                stack.append(actual.right_node)



    
