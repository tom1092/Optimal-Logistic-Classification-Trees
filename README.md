# Optimal-Logistic-Classification-Trees
Repository of the paper OLCT

To run the code properly, there are these few requirements:

Package name | Version
------------ | -------------
[Python](https://www.python.org/) | 3.9.9
[Numpy](http://www.numpy.org/) | 1.22.0
[matplotlib](https://matplotlib.org/) | 3.5.1 
[scikit-learn](https://scikit-learn.org/stable/) | 1.0.2
[scipy](https://scipy.org/) | 1.0.2
[gurobipy](https://www.gurobi.com/) | 9.5.0


## Usage

To replicate all the experiments of the paper you can run the main of each model and follow the instruction of the argparse helper.

Please note that the list of the seed in the main function should change (simply uncomment) to run different experiments.

## Datasets

We did not upload the datasets used in the experiments since they are all publicy available at the UCI [repository](https://archive.ics.uci.edu/datasets).
Our code read each datasets as a .npy file where the first column must be the target (0/1) of the binary classification task.
