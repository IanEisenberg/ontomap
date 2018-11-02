import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

# ******************************************************************************
# Utility functions
# ******************************************************************************
def load_ontology(ontology_file):
    """ loads an ontology pickle file """
    ontology = pickle.load(open(ontology_file, 'rb'))
    return ontology

# ******************************************************************************
# Utility functions for mapping
# ******************************************************************************
def run_linear(data, scores, clf=RidgeCV(fit_intercept=False)):
    """
    Run Knearest neighbor using precomputed distances to create an ontological mapping
    
    Args:
        data: dataframe with variables to reconstruct as columns
        scores: ontological scores
        clf: linear model that returns coefs
    """
    y=scale(data)
    clf.fit(scores, y)

    out = clf.coef_
    if len(out.shape)==1:
        out = out.reshape(1,-1)
    out = pd.DataFrame(out, columns=scores.columns)
    out.index = data.columns
    return out

def KNN_map(data, ontology, ontology_data=None, k=10, weights='distance',
            distance_metric='correlation'):
    """
    Maps variable into an ontology
    
    Performs ontological mapping as described in PAPER
    
    Args:
        data: participant X variable dataset. Columns not included in the
              ontology will be mapped using the rest of the (overlapping)
              variables
        ontology: DV x embedding (e.g. factor loadings) matrix (pandas df). Must overlap
                  with some variable in data
        ontology_data: the data used to create the ontology (pandas df). Used to create a
                       distance matrix to train the KNN regressor. If ontology
                       data is set to None, the data is used to compute the
                       distances
        k: passed to KNeighborsRegressor
        weights: passed to KNeighborsRegressor
        distance_metric: used to compute distances for KNNR
    
    Returns:
        mapping: dataframe with ontology embedding for variables not in the ontology
        neighbors: dictionary with list of k tuples of (neighbor, distance) used for each variable
    """
    # variables to map
    tomap = list(set(data.columns) - set(ontology.index))
    # contextual variables
    overlap = list(set(data.columns) & set(ontology.index))
    # subset/reorder data
    ontology = ontology.loc[overlap]
    data = data.loc[:, tomap+overlap]
    # set up KNN regressor
    if ontology_data is not None:
        ontology_data = ontology_data.loc[:,overlap]
        distances = pd.DataFrame(squareform(pdist(ontology_data.T, metric=distance_metric)), 
                                 index=ontology_data.columns, 
                                 columns=ontology_data.columns)
    else:
        distances = pd.DataFrame(squareform(pdist(data.loc[:,overlap].T, metric=distance_metric)), 
                                 index=overlap, 
                                 columns=overlap)
    clf = KNeighborsRegressor(metric='precomputed', n_neighbors=k, weights=weights)
    clf.fit(distances, ontology)
    
    # test distances
    tomap_distances = pd.DataFrame(squareform(pdist(data.T, metric=distance_metric)), 
                                 index=data.columns, 
                                 columns=data.columns)[tomap].drop(tomap).values
    mapped = pd.DataFrame(clf.predict(tomap_distances.T), index=tomap,
                          columns=ontology.columns)
    # get neighbors
    neighbors = clf.kneighbors(tomap_distances.T)
    neighbor_dict = {}
    for i, v in enumerate(tomap):
        v_neighbors = [(overlap[x], d) for d,x in zip(neighbors[0][i], neighbors[1][i])]
        neighbor_dict[v] = v_neighbors
    return mapped, neighbor_dict

# ******************************************************************************
# Utility functions to calculate factor scores a la self-regulation ontology
# ******************************************************************************    
def transform_remove_skew(data, threshold=1, 
                          positive_skewed=None,
                          negative_skewed=None):
    data = data.copy()
    if positive_skewed is None:
        positive_skewed = data.skew()>threshold
    if negative_skewed is None:
        negative_skewed = data.skew()<-threshold
    positive_subset = data.loc[:,positive_skewed]
    negative_subset = data.loc[:,negative_skewed]
    # transform variables
    # log transform for positive skew
    positive_subset = np.log(positive_subset)
    successful_transforms = positive_subset.loc[:,abs(positive_subset.skew())<threshold]
    dropped_vars = set(positive_subset)-set(successful_transforms)
    # replace transformed variables
    data.drop(positive_subset, axis=1, inplace = True)
    successful_transforms.columns = [i + '.logTr' for i in successful_transforms]
    print('*'*40)
    print('Dropping %s positively skewed data that could not be transformed successfully:' % len(dropped_vars))
    print('\n'.join(dropped_vars))
    print('*'*40)
    data = pd.concat([data, successful_transforms], axis = 1)
    # reflected log transform for negative skew
    negative_subset = np.log(negative_subset.max()+1-negative_subset)
    successful_transforms = negative_subset.loc[:,abs(negative_subset.skew())<threshold]
    dropped_vars = set(negative_subset)-set(successful_transforms)
    # replace transformed variables
    data.drop(negative_subset, axis=1, inplace = True)
    successful_transforms.columns = [i + '.ReflogTr' for i in successful_transforms]
    print('*'*40)
    print('Dropping %s negatively skewed data that could not be transformed successfully:' % len(dropped_vars))
    print('\n'.join(dropped_vars))
    print('*'*40)
    data = pd.concat([data, successful_transforms], axis=1)
    return data.sort_index(axis = 1)
    

def transfer_scores(data, ontology_data, ontology_weights, imputer=None):
    """ calculates factor scores in a new dataset based on a reference results object 
    
    Args:
        data: participant X variable dataset. Columns not included in the
              ontology will be mapped using the rest of the (overlapping)
              variables
        ontology_data: the data (pandas df) used to create the ontology. Used to transform
            the variables in "data" as the original ontology was
        ontology_weights: DV x embedding weight matrix (pandas df), that when multiplied
            with "ontology_data" results in factor scores
            
    Returns:
        mapping: dataframe with ontology embedding for variables not in the ontology
        neighbors: dictionary with list of k tuples of (neighbor, distance) used for each variable
    """
    ref_data = ontology_data
    # transform data
    positive_skewed = [i.replace('.logTr', '') for i in ref_data.columns if ".logTr" in i]
    negative_skewed = [i.replace('.ReflogTr', '') for i in ref_data.columns if ".ReflogTr" in i]
    DVs = [i.replace('.logTr','').replace('.ReflogTr','') for i in ref_data.columns]
    data = data.loc[:, DVs]
    data = transform_remove_skew(data,
                                 positive_skewed=positive_skewed,
                                 negative_skewed=negative_skewed)
    subset = data.loc[:, ontology_weights.index]
    scaled_data = scale(subset)
    # calculate scores
    scores = pd.DataFrame(scaled_data.dot(ontology_weights),
                          index=data.index,
                          columns=ontology_weights.columns)
    return scores
