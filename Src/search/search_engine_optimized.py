import os 
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from numba import njit, prange

# optimized 2
@njit(cache=True)
def compute_calc(np_data, q_vec, sim, n):
    for i in prange(n):
        num = np.linalg.norm(np_data[:, i]) * np.linalg.norm(q_vec)
        if num == 0:
            sim[i] = 0
        else:
            sim[i] = np.dot(np_data[:, i], q_vec) / num
    
    return sim

# def get_similar_articles(q, df, col, vectorizer):
#     # Convert the query become a vector
#     q = [q]
#     q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    
#     # Calculate the similarity
#     np_data=df.values
#     sim = np.zeros((df.shape[1])) 
#     range_n = np.array(range(np.shape(np_data)[1]))
#     sim = compute_calc(np_data, q_vec, sim, range_n)
    
#     # prepare result
#     sim = list(enumerate(sim))
#     sim_sorted = sorted(sim, key=lambda x: x[1], reverse=True)[:10]
#     return sim_sorted

class Search_Engine:
    def __init__(self, cols, cols_weights, df=None, vectorizers={}):
        self.time_measure = 'None'
        self.frequency_uniqueness_avg_measure = 'None'
        self.resulting_simalarities = 'None'
        self.cols = cols
        self.cols_weights = cols_weights
        self.df = df
        if len(vectorizers.keys()) == 0:
            self.get_saved_vectorizers()
        else:
            self.vectorizers = vectorizers
        self.preprocessed_data = self.prepare_processed()

    def get_saved_vectorizers(self, rel_dir='../Results/saved_vectorizers'):
        num_saved_vectorizers = len([name for name in os.listdir(rel_dir)])
        self.vectorizers = {}

        # loading vectorizers
        if num_saved_vectorizers > 0:
            for col in self.cols:
                try:
                    file = open(f"{rel_dir}/{col}_tfidf.pickle", 'rb')
                    self.vectorizers[col] =  pickle.load(file)
                    file.close()
                except:
                    self.vectorizers[col] = None
        else:
            pass
            # print('no vectorizers saved found!')
        
        # print('vectorizers:', self.vectorizers)

    
    def data_preprocessing(self, col, vectorizer):
        # transform it as a vector (tfidf)

        # print(col)
        X = vectorizer.transform(self.df[col])
        X = X.T.toarray() # Convert the X as transposed matrix

        # Create a DataFrame and set the vocabulary as the index
        # SE_data = pd.DataFrame(X, index=vectorizer.get_feature_names())
        return X

    def prepare_processed(self):
        preprocessed_data = {}
        for col in self.cols:
            if self.vectorizers[col] == None:
                continue
            preprocessed_data[col] = self.vectorizers[col].transform(self.df[col]).T.toarray()

        return preprocessed_data


    def get_similar_articles(self, q, data, col, vectorizer):
        # Convert the query become a vector
        q = [q]
        q_vec = vectorizer.transform(q).toarray().reshape(data.shape[0],)
        
        # Calculate the similarity
        np_data = data
        sim = np.zeros((data.shape[1])) 
        range_n = np.array(range(np.shape(np_data)[1]))
        n = np.shape(np_data)[1]
        sim = compute_calc(np_data, q_vec, sim, n)
        
        # prepare result
        sim = list(enumerate(sim))
        sim_sorted = sim
        return sim_sorted


    def overall_search(self, q): 
        def match_func(txt):
            if q in ' '.join(txt):
                return 1
            return 0

        self.time_measure = 0
        self.frequency_uniqueness_avg_measure = 0
        similarities_list = []
        
        # apply search over every col of interest
        num_cols_score = len(self.cols) - 1
        for col in self.cols:
            print(col)
            if col != 'tags':
                start_time = time.time()
                # 1 get vectorizer for corresponding column
                vectorizer = self.vectorizers[col]
                if (vectorizer == None) and (col in self.cols_weights):
                    num_cols_score -= 1
                    del self.cols_weights[col]
                    continue

                print('preprocessing time', end=' ')
                preprocessing_start_time = time.time()
                df_i = self.preprocessed_data[col]
                print((time.time() - preprocessing_start_time) * 10 ** 3)

                # 2- count similarities (each column)
                time_measure = None
                most_freq_measure = None
                sorted_docs_with_scores_content = self.get_similar_articles(q, df_i, col, vectorizer)  # call function

                # 3- results
                vocab_ = vectorizer.vocabulary_
                # print(f"number of unique words: {len(vocab_.keys())}")
                most_freq_word = sorted(vocab_.items(), key=lambda x: x[1], reverse=True)[:1][0]
                # print('most frequent word is --> {} ({} times)'.format(most_freq_word[0], most_freq_word[1]))

                score = len(vocab_.keys()) / most_freq_word[1]
                # time_measure = (time.time() - start_time) * 10**3
                print('Ratio: {:.3f}'.format(score))
                print()
                # print('time measure:', time_measure)
            else:
                most_freq_measure = None  
                start_time = time.time()
                
                # doc_ids = list(self.df['tags'].index)
                # q_list = np.array(q.split(' '))
                # sim_score = list(np.zeros(self.df.shape[0]))

                # for i, tag in enumerate(self.df['tags']):
                #     for str_tag in tag:
                #         q_list_map = np.vectorize(lambda x: x in str_tag)(q_list) 
                #         if True in q_list_map:
                #             sim_score[i] += 1

                # sim_non_sorted = list(zip(doc_ids, sim_score))
                vals = self.df[col].tolist()
                sim_non_sorted = list(map(match_func, vals))
                sim_non_sorted = list(enumerate(sim_non_sorted))
                sorted_docs_with_scores_content = sim_non_sorted
                
                score = 0
                print('Ratio: {:.3f}'.format(score))
                print()
                # print('time measure:', time_measure)
            
            similarities_list.append(sorted_docs_with_scores_content)
            time_measure = (time.time() - start_time) * 10**3
            print('time measure:', time_measure)
            self.time_measure += time_measure
            self.frequency_uniqueness_avg_measure += (score)

            # print('--------------------')
        
        start_time = time.time()
        self.frequency_uniqueness_avg_measure /= num_cols_score

        # print("-" * 25, 'FINAL', '-' * 25)
        # print()
        # print('search engine time taken', self.time_measure, 'ms')
        # print('search engine average score', self.frequency_uniqueness_avg_measure, '(uniqueness/frequency)')
        # print("-" * 25, '-----', '-' * 25) 
        
        averaged_scores_ids = np.array(similarities_list)[0, :, 0]
        averaged_scores = np.average(np.array(similarities_list)[:, :, 1], axis=0, weights=list(self.cols_weights.values()))
        self.resulting_simalarities = list(zip(averaged_scores_ids, averaged_scores))
        time_measure = (time.time() - start_time) * 10**3
        print('last time measure:', time_measure)
        
        return self.resulting_simalarities, self.time_measure, self.frequency_uniqueness_avg_measure

