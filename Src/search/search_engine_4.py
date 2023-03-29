import time
from haystack.utils.cleaning import clean_wiki_text
from haystack.utils import convert_files_to_docs, fetch_archive_from_http

from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline

from haystack.document_stores import ElasticsearchDocumentStore
import pandas as pd

class Search_Engine:
    def __init__(self):
        ELASTIC_PASSWORD = "0K81s_pf_hHC*aiYz1xg"
        document_store = ElasticsearchDocumentStore(host="localhost", port="9200", scheme='https',
                                                    username="elastic", password=ELASTIC_PASSWORD,
                                                    index="se_shai_haystack", ca_certs="../Certs/http_ca.crt")

        retriever = BM25Retriever(document_store=document_store)

        # reader = FARMReader(model_name_or_path="aubmindlab/bert-base-arabertv2", use_gpu=True, context_window_size=1000)
        # self.pipe = ExtractiveQAPipeline(reader, retriever)
        self.pipe = DocumentSearchPipeline(retriever)

        self.df = pd.read_csv('../Data/processed/SE_data4.csv')

    def overall_search(self, q):
        start_time = time.time()

        number_of_answers_to_fetch = 5
        # prediction = self.pipe.run(
        #     query=q, params=
        #     {"Retriever": {"top_k": 20}, "Reader": {"top_k": number_of_answers_to_fetch}}
        # )
        prediction = self.pipe.run(
            query=q, params=
            {"Retriever": {"top_k": number_of_answers_to_fetch}}
        )
        urls = []
        titles = []
        texts = []
        for i in range(len(prediction['documents'])):
            url = prediction['documents'][i].meta['url']
            title = prediction['documents'][i].meta['title']
            text = self.df.loc[prediction['documents'][i].meta['doc_id']]['text']
            urls.append(url)
            titles.append(title)
            texts.append(text)
        
        time_measure = (time.time() - start_time) * 10**3
        print('last time measure:', time_measure)
        
        return urls, titles, texts



