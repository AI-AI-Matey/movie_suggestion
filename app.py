import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask import Flask, request, jsonify, render_template

# Read data from file:
with open("movie_sugg_model/movie_embeddings_dict.pkl", 'rb') as f:
    movie_embeddings_dict = pickle.load(f)
all_movie_names = movie_embeddings_dict.keys()

with open("movie_sugg_model/cast_embeddings_dict.pkl", 'rb') as f:
    cast_emb_dict = pickle.load(f)
all_cast_names = cast_emb_dict.keys()

# All required Functions
def get_did_you_mean(inp_str, name_list_to_check):
    return [ent for ent in name_list_to_check if re.search(inp_str, ent, re.IGNORECASE)]

def get_similars(inp_name, dict_to_consider):
    try:
        inp_name_vec = dict_to_consider[inp_name]
        cos_sim_res = cosine_similarity([inp_name_vec], list(dict_to_consider.values()))

        dict_keys = list(dict_to_consider.keys())

        TOP_N = 20
        top_n_ind = cos_sim_res[0].argsort()[-TOP_N:]
        top_n_similarities = [(str(cos_sim_res[0][ind]), key_) for ind, key_ in enumerate(dict_keys) if ind in top_n_ind]
    
        top_n_similarities = sorted(top_n_similarities, reverse = True)
        return top_n_similarities
    except:
        return None

# app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/movie/<name>')
def suggest_sim_movie(name):
    output_val =  {"input": None, 
                   "suggestion": None,
                   "did_you_mean": None
                  }
    similars_ = get_similars(name, movie_embeddings_dict)
    if similars_:
        output_val["input"] = name
        output_val["suggestion"] = similars_
    else:        
        did_you_mean = get_did_you_mean(inp_str = name, name_list_to_check = all_movie_names)
        output_val["did_you_mean"] = did_you_mean
    return jsonify(results=output_val)

@app.route('/cast/<name>')
def suggest_sim_cast(name):
    output_val =  {"input": None, 
                   "suggestion": None,
                   "did_you_mean": None
                  }
    similars_ = get_similars(name, cast_emb_dict)
    if similars_:
        output_val["input"] = name
        output_val["suggestion"] = similars_
    else:        
        did_you_mean = get_did_you_mean(inp_str = name, name_list_to_check = all_cast_names)
        output_val["did_you_mean"] = did_you_mean
    return jsonify(results=output_val)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

