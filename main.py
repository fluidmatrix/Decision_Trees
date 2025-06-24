import pandas as pd 
import numpy as np

#Manually creating a dataset for the decision tree example
#This dataset is a simple binary classification problem with three features
# namely, ear shape, face shape, and whiskers (all binary features with 0 r 1 #values)
X_train = np.array([[1, 1, 1],
 [0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])
#This is the target variable, where 1 indicates a positive class belonging to cat and 0 indicates a negative class belonging to not cat
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1-p) * np.log2(1 - p)
    
def split_indices(X, feature_index):
    """
    Splits the dataset into two groups based on the feature at feature_index.
    Returns the indices of the two groups.
    """
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[feature_index] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

def weighted_entropy(X, y, left_indices, right_indices):
    """
    Calculates the weighted entropy of the split.
    """
    w_left = len(left_indices) / len(X)
    w_right = len(right_indices) / len(X)
    p_left = sum(y[left_indices]) / len(left_indices)
    p_right = sum(y[right_indices]) / len(right_indices)

    return w_left * entropy(p_left) + w_right * entropy(p_right)

left_indices, right_indices = split_indices(X_train, 0)

def information_gain(X, y, left_indices, right_indices):
    """
    Calculates the information gain of the split.
    """
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy

for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")

"""
Currently, the recursive block still needs to be developed
This is just the functions for the workings of the build_recursive_tree
"""