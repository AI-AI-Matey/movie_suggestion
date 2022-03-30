
import pickle
from flask import Flask, request, jsonify, render_template
from backend.scripts.utils import *
import logging
log = logging.getLogger()
log.setLevel(logging.WARNING)

# Read data from file:
with open("backend/src/movie_embeddings_dict.pkl", 'rb') as f:
    movie_embeddings_dict = pickle.load(f)

all_movie_names = movie_embeddings_dict.keys()

with open("backend/src/cast_embeddings_dict.pkl", 'rb') as f:
    cast_emb_dict = pickle.load(f)

all_cast_names = cast_emb_dict.keys()

# Initiate Flask app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

def get_default_output():

    return {"input": None,
            "suggestion": None,
            "did_you_mean": None
            }


# API Routes
@app.route('/movie/<name>')
def suggest_sim_movie(name):
    
    output_dict = get_default_output()

    try:
        similars_ = get_similar_items(query_name=name, vector_dict=movie_embeddings_dict)
    
        # If results are found
        output_dict["input"] = name
        output_dict["suggestion"] = similars_

        return jsonify(results=output_dict), 200
        
    except KeyError:
        
        # If results are not found
        did_you_mean = search_dict(query=name, check_list=all_movie_names)
        output_dict["did_you_mean"] = did_you_mean

        return jsonify(results=output_dict), 404
    

@app.route('/cast/<name>')
def suggest_sim_cast(name):
    
    output_dict = get_default_output()

    try:
        similars_ = get_similar_items(query_name=name, vector_dict=cast_emb_dict)
    
        # If results are found
        output_dict["input"] = name
        output_dict["suggestion"] = similars_

        return jsonify(results=output_dict), 200
        
    except KeyError:
        
        # If results are not found
        did_you_mean = search_dict(query=name, check_list=all_cast_names)
        output_dict["did_you_mean"] = did_you_mean

        return jsonify(results=output_dict), 404
    

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)

