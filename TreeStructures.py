from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

class TreeNode:
    def __init__(self, id, depth, left_node_id = None, right_node_id = None, left_node = None, right_node = None, feature = None, threshold = None, is_leaf = None, value = None):
        self.id = id
        self.depth = depth
        self.left_node_id = left_node_id
        self.right_node_id = right_node_id
        self.left_node = left_node
        self.right_node = right_node
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
        self.prob = None
        self.impurity = None


    
    @staticmethod
    def copy_node(node):
        new = TreeNode(node.id, node.depth, node.left_node_id, node.right_node_id, node.left_node, node.right_node, node.feature, node.threshold, node.is_leaf, node.value)
        new.parent_id = node.parent_id
        new.data_idxs = node.data_idxs
        new.impurity = node.impurity
        new.weights = node.weights
        new.intercept = node.intercept
        new.w = node.w
        return new


class ClassificationTree:

    def __init__(self, min_samples=None, oblique = False, depth = None, decisor = False):
        self.tree = {}
        self.min_samples = min_samples
        self.depth = depth
        self.n_leaves = 0
        self.oblique = oblique
        self.n_nodes = None

        #This attribute indicate if the tree is a 'decisor' tree
        #i.e. a structure where each branch is a linear classifier (svm/logistic)
        #in these cases the final label is predicted by the last branch rather than by the leaf
        #on which the point fall
        self.decisor = decisor


    #Crea l'albero iniziale in forma di dizionario partendo da un albero con radice root_node
    def initialize(self, data, label, root_node):
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

        #Imposto i padri ogni figlio
        for i in range(len(self.tree)):
            #Verifico se è un branch
            if self.tree[i].left_node_id != self.tree[i].right_node_id:
                #In tal caso setto i relativi padri
                self.tree[self.tree[i].left_node_id].parent_id = i
                self.tree[self.tree[i].right_node_id].parent_id = i
                self.tree[self.tree[i].id].left_node = self.tree[self.tree[i].left_node_id]
                self.tree[self.tree[i].id].right_node = self.tree[self.tree[i].right_node_id]

        #Costruisco indici elementi del dataset associati ad ogni nodo
        self.build_idxs_of_subtree(data, range(len(data)), self.tree[0], oblique = self.oblique)



    #Crea l'albero iniziale random e completo (bilanciato) in forma di dizionario 
    def random_complete_initialize(self, n_features):
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




    #Initialize a Classification tree given a CART structure (sklearn tree object)
    def initialize_from_CART(self, data, label, clf):
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
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
                self.tree[node_id] = TreeNode(node_id, node_depth[node_id], children_left[node_id], children_right[node_id], None, None, feature[node_id], threshold[node_id], False, -1)
                if self.oblique:
                    ej = np.zeros(len(data[0]))
                    ej[feature[node_id]] = 1
                    self.tree[node_id].weights = ej
                    self.tree[node_id].intercept = -threshold[node_id]
            else:
                is_leaves[node_id] = True
                self.tree[node_id] = TreeNode(node_id, node_depth[node_id], -1, -1, None, None, feature[node_id], threshold[node_id], True, np.argmax(value[node_id]))
                self.n_leaves += 1

        #Set father
        for i in range(len(self.tree)):
            #VTest if it is a branch
            if self.tree[i].left_node_id != self.tree[i].right_node_id:
                self.tree[self.tree[i].left_node_id].parent_id = i
                self.tree[self.tree[i].right_node_id].parent_id = i
                self.tree[self.tree[i].id].left_node = self.tree[self.tree[i].left_node_id]
                self.tree[self.tree[i].id].right_node = self.tree[self.tree[i].right_node_id]

        #Build the subsets of indexes given train data of each node
        self.build_idxs_of_subtree(data, range(len(data)), self.tree[0], oblique = self.oblique)
        self.depth = self.get_depth(self.tree[0])


    #Get the depth of the tree
    def get_depth(self, root):
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


    #Predict the label of the point starting from the node 'root'
    def predict_p(self, point, root):
        actual = root
        while(not actual.is_leaf):
            if point[actual.feature] < actual.threshold:
                actual = actual.left_node
            else:
                actual = actual.right_node
        return actual.value
    

    #Return the np array of predicstions for each sample in data starting from the node 'root'
    def predict_data(self, data, root):
        return np.array([self.predict_p(p, root) for p in data])


    
    @staticmethod
    def copy_tree(tree):
        new = ClassificationTree(min_samples=tree.min_samples, oblique = tree.oblique)
        new.depth = tree.depth
        new.n_leaves = tree.n_leaves
        for (node_id, node) in tree.tree.items():
            new.tree[node_id] = TreeNode(node_id, node.depth, node.left_node_id, node.right_node_id, None, None, node.feature, node.threshold, node.is_leaf, node.value)
            new.tree[node_id].parent_id = node.parent_id
            new.tree[node_id].data_idxs = node.data_idxs
            new.tree[node_id].weights = node.weights
            new.tree[node_id].intercept = node.intercept

        #Ora che ho istanziato tutti i nodi vado a settare i puntatori ai figli per ogni nodo
        #Uso una BFS
        stack = [new.tree[0]]
        while stack:
            actual = stack.pop()
            if not actual.is_leaf:
                actual.left_node = new.tree[actual.left_node_id]
                actual.right_node = new.tree[actual.right_node_id]
                stack.append(actual.left_node)
                stack.append(actual.right_node)

        return new


    #Get the list of nodes at a given depth
    @staticmethod
    def get_nodes_at_depth(depth, tree):
        nodes = []
        for (id, node) in tree.tree.items():
            if node.depth == depth:
                nodes.append(node)
        return nodes


    #Print the structure of the tree in a readable way
    def print_tree_structure(self):

        print("The binary tree has %s nodes and has "
              "the following structure:"
              % len(self.tree))

        if self.oblique:
            for i in self.tree.keys():
               
                if self.tree[i].is_leaf:
                    print("%snode=%s is child of node %s. It's a leaf node. Np: %s - Imp: %s - Value: %s" % (self.tree[i].depth * "\t", i, self.tree[i].parent_id, len(self.tree[i].data_idxs), self.tree[i].impurity, self.tree[i].value))
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
                            self.tree[i].intercept,
                            self.tree[i].right_node_id,
                            ))
        else:
            for i in self.tree.keys():
                if self.tree[i].is_leaf:
                    print("%snode=%s is child of node %s. It's a leaf node. Np: %s - Imp: %s - Value: %s" % (self.tree[i].depth * "\t", i, self.tree[i].parent_id, len(self.tree[i].data_idxs), self.tree[i].impurity, self.tree[i].value))
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


    #Build the sets of the indexes of the data points that reach each node in the tree
    def build_idxs_of_subtree(self, data, idxs, root_node, oblique=False):
        
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



    #Starting from the node 'root_node', the method returns the list of the nodes which
    #the point x follows to arrive at a leaf
    @staticmethod
    def get_path_to(x, root_node, oblique):

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


    #Compute the loss of the logistic tree as the summation of the log_loss
    #and the regularization penalty
    def compute_log_loss(self, X, y, regularizer = 1):
        
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
    


    #Perform a refinement of the last branching layer for logistic classification trees
    def refine_last_branch_layer(self, X, y, parallel = False):
    
        #search for each last branch
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
                if not parallel:
                    lr = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = branch.C).fit(X[branch.data_idxs], y[branch.data_idxs])
                    branch.weights = np.squeeze(lr.coef_)
                    branch.intercept = lr.intercept_
                else:

                    #Get the best logistic regression model on a single feature
                    best_weights = None
                    best_loss = np.inf
                    #For each feature
                    for j in range(len(X[0])):
                        lr = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = branch.C).fit(X[branch.data_idxs, j], y[branch.data_idxs])
                        weights = np.zeros(len(X[0]))
                        weights[j] = np.squeeze(lr.coef_)[0]
                        loss = 0
                        for i in range(len(branch.data_idxs)):
                            loss += branch.C * np.log(1+np.exp(-y[branch.data_idxs[i]]*(np.dot(X[branch.data_idxs[i]], weights) + lr.intercept_)))
                        if loss < best_loss:
                            best_loss = loss
                            best_weights = weights
                            best_intercept = lr.intercept_

                    branch.weights = best_weights
                    branch.intercept = best_intercept



    #Get the predicted label for the point x in the case of 'decisor' classification trees
    #i.e. in these structures the prediction is made by the last branch node rather than the leaf
    @staticmethod
    def get_label_decisor_trees(x, root_node):

        actual_node = root_node
        pred = actual_node.value
        while(not actual_node.is_leaf):
            weights = actual_node.weights
            intercept = actual_node.intercept

            if np.dot(x, weights) + intercept <= 0:
                pred = -1
                actual_node = actual_node.left_node
            else:
                pred = 1
                actual_node = actual_node.right_node

        return pred


    #Predice la label degli elementi data nel sottoalbero con radice root_node
    @staticmethod
    def predict_label(data, root_node, oblique, decisor=False):

        if decisor:
            predictions = [ClassificationTree.get_label_decisor_trees(x, root_node) for x in data[:,]]
        else:
            predictions = [root_node.value if root_node.is_leaf else ClassificationTree.get_path_to(x, root_node, oblique)[-1].value for x in data[:,]]
            predictions = [predictions[i] if predictions[i]!=None else 0 for i in range(len(predictions))]
        return predictions



    #Restituisce id della foglia del sottoalbero con radice in root_node che predice x
    @staticmethod
    def predict_leaf(x, root_node, oblique):
        path = ClassificationTree.get_path_to(x, root_node, oblique)
        return path[-1].id



    #Restituisce la loss del sottoalbero con radice in root_node
    @staticmethod
    def misclassification_loss(root_node, data, target, indexes, oblique=False, decisor = False):
        if len(indexes) > 0:
            preds = ClassificationTree.predict_label(data[indexes], root_node, oblique, decisor)
            n_misclassified = np.count_nonzero(target[indexes]-preds)
            return n_misclassified/len(indexes)
        else:
            return 0



    #Restore the dictionary structure of the tree using a DFS
    @staticmethod
    def restore_tree(tree, X, y):
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
        
    

    #Get leaves and branch nodes
    @staticmethod
    def get_leaves_and_branches(root):
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



    #Create two new leaves of the node and set their labels by majority
    @staticmethod
    def create_new_children(node, X, y, max_id, feature, threshold, oblique = False, weights=None, intercept=None):

        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        if oblique:
            node.weights = weights
            node.intercept = intercept
        id_left = max_id+1
        id_right = max_id+2
        left_child_node = TreeNode(id_left, node.depth+1, -1, -1, None, None, None, None, True, None)
        right_child_node = TreeNode(id_right, node.depth+1, -1, -1, None, None, None, None, True, None)
        left_child_node.parent_id = node.id
        right_child_node.parent_id = node.id
        node.left_node_id = id_left
        node.right_node_id = id_right
        node.left_node = left_child_node
        node.right_node = right_child_node

        ClassificationTree.build_idxs_of_subtree(X, node.data_idxs, node, oblique)
        bins = np.bincount(y[left_child_node.data_idxs])
        best_class_left = -1
        best_class_right = -1
        if len(bins > 0):
            best_class_left = bins.argmax()
        bins = np.bincount(y[right_child_node.data_idxs])
        if len(bins > 0):
            best_class_right = bins.argmax()
        left_child_node.value = best_class_left
        right_child_node.value = best_class_right


    #Delete a given node from the tree
    @staticmethod
    def delete_node(node_id, tree):
        T = tree.tree
        stack = [node_id]
        while (len(stack) > 0):
            actual_node = stack.pop()
            if not T[actual_node].is_leaf:
                stack.append(T[actual_node].left_node_id)
                stack.append(T[actual_node].right_node_id)
            T.pop(actual_node)


    #Replace the sub tree rooted at node_A with the one rooted at node_B
    @staticmethod
    def replace_node(node_A, node_B, tree):
        tree = tree.tree

        #Save id of the father
        parent_A_id = node_A.parent_id
        if parent_A_id != -1:
            parent_A = tree[parent_A_id]
            #need to understand if a was lef or right child
            if parent_A.left_node_id == node_A.id:
                parent_A.left_node_id = node_B.id
                parent_A.left_node = node_B

            elif parent_A.right_node_id == node_A.id:
                parent_A.right_node_id = node_B.id
                parent_A.right_node = node_B

        node_B.parent_id = parent_A_id

        #reset the depth of each node of the subtree rooted at node_B
        node_B.depth = node_A.depth
        stack = [node_B]
        while (len(stack) > 0):
            actual_node = stack.pop()
            if not actual_node.is_leaf:
                actual_node.left_node.depth = actual_node.depth + 1
                actual_node.right_node.depth = actual_node.depth + 1
                stack.append(actual_node.left_node)
                stack.append(actual_node.right_node)



    #Get the list of the features used by each node of the tree
    @staticmethod
    def get_features(T):
        features = []
        stack = [T.tree[0]]
        while stack:
            actual = stack.pop()
            if not actual.is_leaf:
                features.append(actual.feature)
                stack.append(actual.left_node)
                stack.append(actual.right_node)
        return features


    #Compute the positive probabilities for each leaf
    def compute_prob(self, X, labels):
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



    #Get the prob to be predicted as positive for the given point
    def predict_prob(self, point):
        leaf = self.tree[self.predict_leaf(point, self.tree[0], self.oblique)]
        return leaf.prob
