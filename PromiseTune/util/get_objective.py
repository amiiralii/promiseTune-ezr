import random
import numpy as np
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler

def get_objective_direct(dict_search, best_solution):
    return dict_search.get(tuple(best_solution)) 

def get_objective(dict_search, best_solution):
    tmp = dict_search.get(tuple(best_solution))
    if tmp:
        return tmp, list(best_solution)
    keys_list = [list(k) for k in dict_search.keys()]
    scaler = MinMaxScaler()
    scaler.fit(keys_list)
    keys_vectors = scaler.transform(keys_list)
    query_vect = scaler.transform([best_solution])[0]
    random.shuffle(keys_vectors)
    kdtree = spatial.KDTree(keys_vectors)
    _, idx = kdtree.query(query_vect, k=1)
    result = list(scaler.inverse_transform([keys_vectors[idx]])[0])
    result = [round(x) for x in result]
    tmp_value = dict_search.get(tuple(result))
    if tmp_value:
        return tmp_value, result
    min_dist = float("inf")
    final_result = None
    final_value = None
    for key in dict_search.keys():
        key_vect = scaler.transform([list(key)])[0]
        dist = np.linalg.norm(key_vect - query_vect)
        if dist < min_dist:
            min_dist = dist
            final_value = dict_search.get(tuple(key))
            final_result = list(key)
    return final_value, final_result

def distance(p1,p2):
	p1 = np.array(p1)
	p2 = np.array(p2)
	dist_orig = np.sum(np.square(p1-p2))
	return dist_orig