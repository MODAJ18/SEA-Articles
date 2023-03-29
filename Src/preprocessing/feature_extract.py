from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def create_vectorizer(df, from_feature='text_clean'):
    # Instantiate a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    if (df[from_feature].unique()[0] == "") & (len(list((df[from_feature].unique()))) == 1):
        return None
    else:
        vectorizer.fit(df[from_feature])
    
    return vectorizer 