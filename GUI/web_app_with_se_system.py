from flask import Flask, request, render_template

import sys
sys.path.append('../Src')

from preprocessing.data_clean import clean_text2
from search.search_engine_4 import Search_Engine
import time

# defined on website start
cols = ['text_clean', 'summary_clean', 'title_clean', 'tags']
cols_weights = {'text_clean': 0.5, 'summary_clean': 0.15, 'title_clean': 0.2, 'tags': 0.1}
SE_ob = Search_Engine()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    search_param = request.args.get("search")
    if request.method == 'POST':
        pass
    
    if search_param:
        overall_time = time.time()

        query_test = str(search_param)
        query_test_cleaned = clean_text2(query_test)
        urls, titles, texts = SE_ob.overall_search(query_test_cleaned)

        SE_time_duration = (time.time() - overall_time) * 10**3
        print(f"total search time duration --> {SE_time_duration} ms")
        return render_template('Search-Results.html', urls=urls, titles=titles, texts=texts)

    return render_template('Search.html')



