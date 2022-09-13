import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

from scipy.special import softmax


def n_cylinders(word):
    if word in ['Single cylinder']:
        return 1
    elif word in ['V2', 'Twin', 'Two cylinder boxer']:
        return 2
    elif word in ['In-line three', 'Diesel']:
        return 3
    elif word in ['V4', 'In-line four', 'Four cylinder boxer']:
        return 4
    elif word in ['In-line six', 'Six cylinder boxer']:
        return 6
    elif word in ['Electric']:
        return 0

def random_rows(df, n_indices):
    indices = df.index.tolist()
    if len(indices) <= n_indices:
        return df.copy()
    rnd_indices = np.random.choice(indices, size=n_indices, replace=False)
    rnd_rows = df.loc[rnd_indices,:]
    return rnd_rows

def binarize_dual(feature):
    """Receives a feature column with binary labels (only two classes)
    Returns a single feature column of zeros and ones and the name of the class encoded with 1 (as a list)"""
    unique_vals = feature.unique()
    cat = unique_vals[0]
    bin_array = np.zeros((len(feature), ))
    bin_array[np.where(feature==cat)] = 1
    bin_array[np.where(feature!=cat)] = 0
    return bin_array, [cat]

def binarize_categorical(feature):
    """Receives a vector of a categorical feature.
    Returns the matrix of the binarized feature and the names of each category
    (which are the columns in the matrix). If the feature has just two classes,
    return one binary column"""
    unique_vals = feature.unique()
    if len(unique_vals) == 2:
        bin_array, unique_vals = binarize_dual(feature)
    else:
        bin_array = np.zeros((len(feature), len(unique_vals)))
        for j, cat in enumerate(unique_vals):
            bin_array[np.where(feature==cat),j] = 1
            bin_array[np.where(feature!=cat),j] = 0
    return bin_array, unique_vals

def n_cylinders(word):
    if word in ['Single cylinder']:
        return 1
    elif word in ['V2', 'Twin', 'Two cylinder boxer']:
        return 2
    elif word in ['In-line three', 'Diesel']:
        return 3
    elif word in ['V4', 'In-line four', 'Four cylinder boxer']:
        return 4
    elif word in ['In-line six', 'Six cylinder boxer']:
        return 6
    elif word in ['Electric']:
        return 0

def feature_to_binary(df, col_name):
    """Create binary vectors corresponding to each possible category in the feature
    represented by the column passed.
    Return the received DataFrame, with the new columns which are the binary arrays"""
    feature = df.loc[:,col_name]
    bin_array, names = binarize_categorical(feature)
    names = [col_name+'_'+str(n) for n in names]
    if len(names) > 1:
        bin_array_t = bin_array.T
        for i, col_t in enumerate(bin_array_t):
            df.loc[:,names[i]] = col_t.T
    else:
        df.loc[:,names[0]] = bin_array.T
    return df

