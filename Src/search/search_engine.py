import os 
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from numba import njit


class Search_Engine:
    def __init__(self, cols, cols_weights, df=None, vectorizers={}):
        self.vectorizers = vectorizers
        self.time_measure = 'None'
        self.frequency_uniqueness_avg_measure = 'None'
        self.resulting_simalarities = 'None'
        self.cols = cols
        self.cols_weights = cols_weights
        self.df = df

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
        SE_data = pd.DataFrame(X, index=vectorizer.get_feature_names())
        return SE_data

    def calc_sim(self, cols):
        sim[i] = np.dot(data.loc[:, i].values, q_vec) / np.linalg.norm(data.loc[:, i]) * np.linalg.norm(q_vec)
        if np.isnan(sim[i]):
            sim[i] = 0
        return sim[i]

    def get_similar_articles(self, q, data, col, vectorizer):
        # Convert the query become a vector
        q = [q]
        q_vec = self.vectorizers[col].transform (q).toarray().reshape(data.shape[0],)
        
        # Calculate the similarity
        sim = {}
        cols = list(range(data.shape[1]))
        for i in range(data.shape[1]):
            sim[i] = np.dot(data.loc[:, i].values, q_vec) / np.linalg.norm(data.loc[:, i]) * np.linalg.norm(q_vec)
            if np.isnan(sim[i]):
                sim[i] = 0

        # Sort the values 
        sim_sorted = list(sim.items())
        return sim_sorted

    def overall_search(self, q): 
        self.time_measure = 0
        self.frequency_uniqueness_avg_measure = 0
        similarities_list = []
        
        # apply search over every col of interest
        num_cols_score = len(self.cols) - 1
        for col in self.cols:
            print(col)
            if col != 'tags':
                # 1 get vectorizer for corresponding column
                vectorizer = self.vectorizers[col]
                if vectorizer == None:
                    num_cols_score -= 1
                    del self.cols_weights[col]
                    continue
                df_i = self.data_preprocessing(col, vectorizer)

                # 2- count similarities (each column)
                time_measure = None
                most_freq_measure = None  
                start_time = time.time()
                sorted_docs_with_scores_content = self.get_similar_articles(q, df_i, col, vectorizer)  # call function
                time_measure = (time.time() - start_time) * 10**3

                # 3- results
                vocab_ = vectorizer.vocabulary_
                # print(f"number of unique words: {len(vocab_.keys())}")
                most_freq_word = sorted(vocab_.items(), key=lambda x: x[1], reverse=True)[:1][0]
                # print('most frequent word is --> {} ({} times)'.format(most_freq_word[0], most_freq_word[1]))

                score = len(vocab_.keys()) / most_freq_word[1]
                print('Ratio: {:.3f}'.format(score))
                print()
                print('time measure:', time_measure)
            else:
                time_measure = None
                most_freq_measure = None  
                start_time = time.time()
                
                doc_ids = list(self.df['tags'].index)
                q_list = np.array(q.split(' '))
                sim_score = list(np.zeros(self.df.shape[0]))

                for i, tag in enumerate(self.df['tags']):
                    for str_tag in tag:
                        q_list_map = np.vectorize(lambda x: x in str_tag)(q_list) 
                        if True in q_list_map:
                            sim_score[i] += 1

                sim_non_sorted = list(zip(doc_ids, sim_score))
                sorted_docs_with_scores_content = sim_non_sorted
                
                time_measure = (time.time() - start_time) * 10**3
                score = 0
                print('Ratio: {:.3f}'.format(score))
                print()
                print('time measure:', time_measure)
                
            similarities_list.append(sorted_docs_with_scores_content)
            self.time_measure += time_measure
            self.frequency_uniqueness_avg_measure += (score)
            # print('--------------------')
        self.frequency_uniqueness_avg_measure /= num_cols_score

        # print("-" * 25, 'FINAL', '-' * 25)
        # print()
        # print('search engine time taken', self.time_measure, 'ms')
        # print('search engine average score', self.frequency_uniqueness_avg_measure, '(uniqueness/frequency)')
        # print("-" * 25, '-----', '-' * 25) 
        
        averaged_scores_ids = np.array(similarities_list)[0, :, 0]
        averaged_scores = np.average(np.array(similarities_list)[:, :, 1], axis=0, weights=list(self.cols_weights.values()))
        self.resulting_simalarities = list(zip(averaged_scores_ids, averaged_scores))
        
        return self.resulting_simalarities, self.time_measure, self.frequency_uniqueness_avg_measure

