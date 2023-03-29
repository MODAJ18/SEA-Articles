## Helper Functions
import re
from nltk.corpus import stopwords
from camel_tools.utils.dediac import dediac_ar # dediacritization tool
from camel_tools.utils.normalize import normalize_alef_maksura_ar # Reducing Orthographic Ambiguity
from camel_tools.utils.normalize import normalize_alef_ar # Reducing Orthographic Ambiguity
from camel_tools.utils.normalize import normalize_teh_marbuta_ar # Reducing Orthographic Ambiguity
from camel_tools.tokenizers.word import simple_word_tokenize # toknenization
from camel_tools.disambig.mle import MLEDisambiguator # Maximum Likelihood Disambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer # tokenization / lemmatization
from bs4 import BeautifulSoup


def remove_urls(text):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ', text)

def remove_html(text):
    return BeautifulSoup(text, "html.parser").text

symb_re = re.compile(r"""[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~،؟…«“\":\"…”]""")
def remove_symbols(text: str) -> str:
    return symb_re.sub(repl="", string=text)

multiple_space_re = re.compile("\s{2,}")
def remove_multiple_whitespace(text):
    return multiple_space_re.sub(repl=" ", string=text)


def clean_text(txt): 
    #remove punctuations
    translator = str.maketrans('', '', punctuations)
    txt = txt.translate(translator)
    
    # remove Tashkeel
    txt = re.sub(arabic_diacritics, '', txt)
    
    # remove longation
    txt = re.sub("[إأآا]", "ا", txt)
    txt = re.sub("ى", "ي", txt)
    txt = re.sub("ؤ", "ء", txt)
    txt = re.sub("ئ", "ء", txt)
    txt = re.sub("ة", "ه", txt)
    txt = re.sub("گ", "ك", txt)
    
    # remove stopwords
    txt = ' '.join(word for word in txt.split() if word not in stop_words)
    
    return txt

def clean_text2(txt):
    txt = remove_urls(txt)
    txt = remove_html(txt)

    # remove stopwords
    # txt = ' '.join(word for word in txt.split() if word not in stop_word_list)

    # dediacritization
    txt = dediac_ar(txt)

    # normalization: Reduce Orthographic Ambiguity and Dialectal Variation
    txt = normalize_alef_maksura_ar(txt)
    txt = normalize_alef_ar(txt)
    txt = normalize_teh_marbuta_ar(txt)

    # normalization: Reducing Morphological Variation
    tokens = simple_word_tokenize(txt)
    disambig = mle.disambiguate(tokens)
    lemmas = [d.analyses[0].analysis['lex'] for d in disambig]
    tokens = tokenizer.tokenize(lemmas)
    txt = ' '.join(tokens)

    # remove longation
    # txt = re.sub("[إأآا]", "ا", txt)
    txt = re.sub("ى", "ي", txt)
    # txt = re.sub("ؤ", "ء", txt)
    # txt = re.sub("ئ", "ء", txt)
    txt = re.sub("ة", "ه", txt)
    # txt = re.sub("گ", "ك", txt)

    # remove non-arabic words, or non-numbers, or non-english words in the text
    txt = re.sub(
        r'[^a-zA-Z\s0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD.0-9]+'
        , ' ', txt)

    # remove symbols
    txt = remove_symbols(txt)

    # remove multiple whitespace
    txt = remove_multiple_whitespace(txt)

    return txt




## Main Functions
# 1
import re
from nltk.corpus import stopwords
import string
# punctuation symbols
punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + \
                                            string.punctuation
# Arabic stop words with nltk
stop_words = stopwords.words()
arabic_diacritics = re.compile("""
                            ّ    | # Shadda
                            َ    | # Fatha
                            ً    | # Tanwin Fath
                            ُ    | # Damma
                            ٌ    | # Tanwin Damm
                            ِ    | # Kasra
                            ٍ    | # Tanwin Kasr
                            ْ    | # Sukun
                            ـ     # Tatwil/Kashida
                        """, re.VERBOSE) 
def clean_data(df):
    # 1 Prepraing Data
    df['text'] = df['content'].apply(lambda x: " ".join(x))

    # 2 removing non-relevant data
    df = df.drop(index=df[(df['content'].str.len() == 0) & (df['title'] == '')].index, axis=0)
    df = df.reset_index(drop=True)
    df = df.fillna('')

    # 3 Text Cleaning
    df['text_clean'] = df['text'].apply(clean_text)
    df['summary_clean'] = df['summary'].apply(clean_text)
    df['title_clean'] = df['title'].apply(clean_text)

    return df


# 2
mle = MLEDisambiguator.pretrained() # instantiation fo MLE disambiguator
stop_words = stopwords.words()
tokenizer = MorphologicalTokenizer(mle, scheme='atbtok', diac=False) # atbtok scheme 
def clean_data2(df):
    # 1 Prepraing Data
    df['text'] = df['content'].apply(lambda x: " ".join(x))

    # 2 removing non-relevant data
    df = df.drop(index=df[(df['content'].str.len() == 0) & (df['title'] == '')].index, axis=0)
    df = df.reset_index(drop=True)
    df = df.fillna('')

    # 3 Text Cleaning
    df['text_clean'] = df['text'].apply(clean_text2)
    df['summary_clean'] = df['summary'].apply(clean_text2)
    df['title_clean'] = df['title'].apply(clean_text2)
    
    return df
