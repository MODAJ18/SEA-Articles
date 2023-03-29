from flask import Flask, request, render_template
import os
import sys
sys.path.append('../Src')

import pickle
import numpy as np
import pandas as pd

from preprocessing.data_prep import read_data
from preprocessing.data_clean import clean_data, clean_text
from preprocessing.feature_extract import create_vectorizer

# CHANGED
# from search.search_engine import Search_Engine
from search.search_engine_optimized2 import Search_Engine

import time


# variables set
data_path = "../Data/husna.json"
cols = ['text_clean', 'summary_clean', 'title_clean', 'tags']
cols_weights = {'text_clean': 0.5, 'summary_clean': 0.15, 'title_clean': 0.2, 'tags': 0.1}
SE_df = pd.read_csv('../Data/processed/SE_data1.csv', keep_default_na=False)
SE_ob = Search_Engine(cols, cols_weights, df=SE_df)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # cols = ['text_clean', 'summary_clean', 'title_clean', 'tags']
    # cols_weights = {'text_clean': 0.5, 'summary_clean': 0.15, 'title_clean': 0.2, 'tags': 0.1}
    # SE_ob = Search_Engine(cols, cols_weights, df=SE_df)
    # SE_ob.get_saved_vectorizers(rel_dir='../Results/saved_vectorizers')
    # results = SE_ob.overall_search(query_test_cleaned)
    search_param = request.args.get("search")
    if request.method == 'POST':
        pass
        # # check if the post request has the file part
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
    
    if search_param:
        overall_time = time.time()

        query_test = str(search_param)
        query_test_cleaned = clean_text(query_test)
        SE_time_duration1 = (time.time() - overall_time) * 10**3
        print(f"time duration 1 --> {SE_time_duration1} ms", end='\n\n')

        overall_time = time.time()
        results = SE_ob.overall_search(query_test_cleaned)
        SE_time_duration2 = (time.time() - overall_time) * 10**3
        print(f"time duration 2 --> {SE_time_duration2} ms", end='\n\n')

        overall_time = time.time()
        top10_results = sorted(results[0], key=lambda x: x[1], reverse=True)[:10]
        SE_time = results[1]
        SE_score = results[2]
        ids = np.array(top10_results, dtype="int32")[:, 0]
        urls = list(SE_df.loc[ids, :]['url'])
        titles = list(SE_df.loc[ids, :]['title'])
        texts = list(SE_df.loc[ids, :]['text'])
        SE_time_duration3 = (time.time() - overall_time) * 10**3
        print(f"time duration 3 --> {SE_time_duration3} ms", end='\n\n\n')
        print()
        print(f"SE Search Time: {SE_time_duration1 + SE_time_duration2 + SE_time_duration3}")

        # # {% for result in results %}
        # #     <tr>
        # #         <td>{{ result[0] }}</td>
        # #         <td>{{ result[1] }}</td>
        # #         <td>{{ result[2] }}</td>
        # #     </tr>
        # # {% endfor %}

        return render_template('Search-Results.html', urls=urls, titles=titles, texts=texts,
                                    SE_time=SE_time, SE_score=SE_score)
        # print(f"time duration {SE_time_duration} ms")
        # return render_template('Search-Results.html')

    # return render_template('main.html', tenure_chart=tenure_chart, box_plot=box_plot,
    #     heat_map=heat_map, CA_scatter_plot=CA_scatter_plot, CA_tsne_plot=CA_tsne_plot,
    #     CA_dist_plot=CA_dist_plot)

    return render_template('Search.html')



