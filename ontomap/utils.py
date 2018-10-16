import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

def run_linear(scores, test_vars, clf=LinearRegression(fit_intercept=False)):
    """
    Run Knearest neighbor using precomputed distances to create an ontological mapping
    
    Args:
        scores: ontological scores
        test_vars: variable to reconstruct
        clf: linear model that returns coefs
    """
    clf.fit(scores, scale(test_vars))
    out = clf.coef_
    if len(out.shape)==1:
        out = out.reshape(1,-1)
    out = pd.DataFrame(out, columns=scores.columns)
    out['var'] = test_vars.columns
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
        ontology: DV x embedding (e.g. factor loadings) matrix. Must overlap
                  with some variable in data
        ontology_data: the data used to create the ontology. Used to create a
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

    
    
