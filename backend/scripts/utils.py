
import re
import logging
from sklearn.metrics.pairwise import cosine_similarity

def get_vector(name, vector_dict):
    """
    Returns a float vector, if present"""

    try:
        return vector_dict[name]
    
    except KeyError as ke:
        logging.debug("DID YOU MEAN?\n", search_dict(query=name, check_list=list(vector_dict.keys())))

        raise KeyError(str(ke))


def get_cos_sim(q, v):
    """
    Returns cosine similarity pair-wise
    
    Expects `q` as Array(float), `v` as Array(Array(float))"""

    return cosine_similarity([q], v)[0]
 

def get_top_indices(x: list, TOP_N: int):
    """
    Returns at max TOP_N indices by sorting in descending order"""

    return x.argsort()[-TOP_N:][::-1]


def get_similar_items(query_name: str, vector_dict: dict, TOP_N=20)-> list:
    """
    Slice TOP N similar vectors and its keys"""

    query_vec = get_vector(name=query_name, vector_dict=vector_dict)

    dict_keys = list(vector_dict.keys())
    dict_values = list(vector_dict.values())

    cos_sim_res = get_cos_sim(q=query_vec, v=dict_values)
    logging.debug(cos_sim_res)    
    
    top_n_ind = get_top_indices(x=cos_sim_res, TOP_N=TOP_N)
    logging.debug(top_n_ind)

    top_n_similarities = [key_ for ind, key_ in enumerate(dict_keys) if ind in top_n_ind]
    top_n_scores = cos_sim_res[top_n_ind].tolist()

    out = list(zip(top_n_scores, top_n_similarities))

    return out


def search_dict(query: str, check_list: list)-> list:
    """
    Acts like 'Did You Mean?' for a given query if exact match is not found.
    
    This function returns all the close search results based on the query"""

    out = []

    for val in check_list:

        if re.search(query, val, re.IGNORECASE):
            out.append(val)

    return out


def make_score_simpler(score: float)-> str:
    
    perc = score * 100
    rounded = int(round(perc, 1))
    
    str_out = str(rounded)+"%"
    
    return str_out
