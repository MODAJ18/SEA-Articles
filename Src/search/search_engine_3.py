import time
from elasticsearch import Elasticsearch

class Search_Engine:
    def __init__(self, cols=None, cols_weights=None, df=None, vectorizers={}):
        # self.vectorizers = vectorizers
        # self.time_measure = 'None'
        # self.frequency_uniqueness_avg_measure = 'None'
        # self.resulting_simalarities = 'None'
        # self.cols = cols
        # self.cols_weights = cols_weights
        # self.df = df
        self.es = None
        self.connect_elasticsearch()

    def connect_elasticsearch(self):
        _es = None
        ELASTIC_PASSWORD = "zJXerPHeN7PEmq5zWRuZ"
        _es = Elasticsearch("https://localhost:9200", 
                    ca_certs='../Certs/http_ca.crt',
                    basic_auth=("elastic", ELASTIC_PASSWORD))
        if _es.ping():
            print('Connected')
            self.es = _es
        else:
            print('Could not Connect')
        return _es

    def overall_search(self, q): 
        start_time = time.time()

        # search in elasticsearch DB
        query = {"multi_match": 
                  { 
                      "query":  q, 
                      "fields": [ "title", "text", "tags"] 
                  }
        }
        resp = self.es.search(index="se_shai", query=query)

        # store top results
        urls = []
        titles = []
        texts = []
        tags = []
        for hit in resp['hits']['hits']:
            urls.append(hit['_source']['url'])
            titles.append(hit['_source']['title'])
            texts.append(hit['_source']['text'])
            tags.append(hit['_source']['tags'])
        urls = urls[:10]
        titles = titles[:10]
        texts = texts[:10]
        tags = tags[:10]
        
        time_measure = (time.time() - start_time) * 10**3
        print('last time measure:', time_measure)
        
        return urls, titles, texts

