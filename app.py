
# coding: utf-8

# In[8]:


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


# In[5]:


def get_similars(inp_name, dict_to_consider):
    inp_name_vec = dict_to_consider[inp_name]
    cos_sim_res = cosine_similarity([inp_name_vec], list(dict_to_consider.values()))
    
    dict_keys = list(dict_to_consider.keys())
    
    TOP_N = 20
    top_n_ind = cos_sim_res[0].argsort()[-TOP_N:]
    top_n_similarities = [key_ for ind, key_ in enumerate(dict_keys) if ind in top_n_ind]
    top_n_similarities.reverse()
    return top_n_similarities


# In[6]:


################ MOVIE RESULTS ################


# In[7]:


# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    
    INP_MOVIE = data["movie"]
    output_val = ("*****\nInput Movie:\n", INP_MOVIE, 
      "\n*****\nThe Movies you may like\n\n", "\n".join(get_similars(INP_MOVIE, movie_embeddings_dict)))
    # send back to browser
    # return data
    return jsonify(results=output_val)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

