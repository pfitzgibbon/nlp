# general libraries
import pandas as pd
import numpy as np
import seaborn as sns
import re
import sys

# sklearn libraries
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# nltk libraries
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.wsd import lesk

# gensim libraries
#from gensim.test.utils import common_texts, get_tmpfile
#from gensim.models import Word2Vec

############### below links read and used to build the following code ############################
# https://python.developreference.com/article/12873061/Clustering+synonym+words+using+NLTK+and+Wordnet
# https://stackoverflow.com/questions/47757435/clustering-synonym-words-using-nltk-and-wordnet
# http://www.nltk.org/howto/wsd.html, word sense disambiguation (WSD)
# https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
# https://towardsdatascience.com/k-means-clustering-chardonnay-reviews-using-scikit-learn-nltk-9df3c59527f3
# http://www.nltk.org/howto/wsd.html
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html

############### below links read but not used, not attempting to use gensim here #################
# https://stackoverflow.com/questions/11798389/what-nlp-tools-to-use-to-match-phrases-having-similar-meaning-or-semantics
# https://radimrehurek.com/gensim/models/word2vec.html

# clean word data
def word_clean(dframe, columnname):
    for w in range(len(dframe[columnname])):
        dframe[columnname][w] = dframe[columnname][w].lower()
        # remove punctuation
        dframe[columnname][w] = re.sub("[^\w\d'\s]+", ' ', dframe[columnname][w])
        # remove tags
        #dframe[columnname][w] = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", dframe[columnname][w])
        # remove special characters and digits
        #dframe[columnname][w] = re.sub("(\\d|\\W)+",' ', dframe[columnname][w])
    return dframe

def nltk_tag_to_wordnet_tag(nltk_tag):
    # read in nltk tag and translate it to wordnet tag 
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def add_to_wordbag(word, tag, sense, use_syn = True, use_hyper = True, use_hypo = True, 
                   use_hol = True, use_part_mer = True, use_sub_mer = True):
    added_words = []
    if use_syn == True: # add synonyms
        try:
            added_words.append(wordnet.synsets(word+'.'+tag+'.'+sense).lemma_names())
        except:
            pass
    if use_hyper == True: # hypernyms are ex: musical instrument to guitar
        try:
            added_words.append(wordnet.synset(word+'.'+tag+'.'+sense).hypernyms())
        except:
            pass
    if use_hol== True: # holonyms are ex: face to eye
        try:
            added_words.append(wordnet.synset(word+'.'+tag+'.'+sense).holonyms())
        except:
            pass
    if use_hypo== True: # hyponym are ex: lasagna to pasta
        try:
            added_words.append(wordnet.synset(word+'.'+tag+'.'+sense).hyponyms())
        except:
            pass
    if use_part_mer == True: # part meronyms, ex: branch for tree
        try:
            added_words.append(wordnet.synset(word+'.'+tag+'.'+sense).part_meronyms())
        except:
            pass
    if use_sub_mer == True: # substance meronyms, ex: maple tree for tree
        try:
            added_words.append(wordnet.synset(word+'.'+tag+'.'+sense).substance_meronyms())
        except:
            pass
    return added_words

def tag_word(sent):
    # read words in sentence
    sent_tagged = []
    for word in sent:
        # use wordnet wsd using lesk to get tags and sense
        word_wsd = str(lesk(sent, word))
        if ('None' in word_wsd) or (len(word) < 3): #WSD doesn't do well on short words
            # tokenize the sentence and find the POS tag for each  using nltk
            nltk_tagged = nltk.pos_tag([word])
            nltk_tagged = [list(elem) for elem in nltk_tagged][0]
            wordnet_tagged = [nltk_tagged[0], nltk_tag_to_wordnet_tag(nltk_tagged[1]), '.01']
        else:
            wordnet_tagged = (word_wsd[word_wsd.find("('")+2:word_wsd.find("')")]).replace('_',' ').split(".")
        sent_tagged.append(wordnet_tagged)
    return sent_tagged
    
def process_sentence(sentence, add_words = True, use_syn = True, use_hyper = True, use_hypo = True, 
                   use_hol = True, use_part_mer = True, use_sub_mer = True):
    lem = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    sent = nltk.word_tokenize(sentence)
    
    # read words in sentence
    sent_tagged = tag_word(sent)
    
    # process the words in the sentence, lemmatize and add related words
    processed_sentence = []
    for word, tag, sense in sent_tagged:
        # perform pruning step, remove stopwords, short words
        if (not word in stop_words) and (len(word) >2):
            if tag is None:
                # if there is no available tag, append the token as is
                processed_sentence.append(word)
            else:        
                # else use the tag to lemmatize the token
                processed_sentence.append(lem.lemmatize(word, tag))
            if add_words == True: #adding too many words can make items artificially more similar
                # add words to the wordbag
                if tag is not None:    
                    processed_sentence.append(add_to_wordbag(word, tag, sense, use_syn = use_syn, 
                        use_hyper = use_hyper, use_hypo = use_hypo, use_hol = use_hol, 
                        use_part_mer = use_part_mer, use_sub_mer = use_sub_mer))
    
    # clean up the data again
    proc_sen = " ".join(str(v) for v in processed_sentence)
    # remove tags
    proc_sen = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", proc_sen)
    # remove punctuation
    proc_sen = re.sub('[^a-zA-Z]', ' ', proc_sen)
    # remove special characters and digits
    proc_sen = re.sub("(\\d|\\W)+"," ", proc_sen)
    # remove added words fluffer
    proc_sen = re.sub("Synset"," ", proc_sen)
    proc_sen = ' '.join( [w for w in proc_sen.split() if len(w)>1] )
    
    return proc_sen

def tfidf_lsa(descriptions):
    #TF-IDF vectorizer approach. TF-IDF, TfidfVectorizer, uses a 
    #in-memory vocabulary (a python dict) to map the most frequent words to features indices 
    #and hence compute a word occurrence frequency (sparse) matrix. The word frequencies are 
    #then reweighted using the Inverse Document Frequency (IDF) vector collected feature-wise 
    #over the corpus. so this does not account for synonyms of the words
    #latent semantic analysis can also be used to reduce dimensionality and discover 
    #latent patterns in the data.
    
    stop_words = stopwords.words('english')
    tfv = TfidfVectorizer(input='content', stop_words = stop_words, ngram_range = (1,1), 
                          max_df=0.5, min_df=2, use_idf=True)
    svd = TruncatedSVD(100) #For LSA, a value of 100 is recommended
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    
    #vectorize the data
    vec_desc = tfv.fit_transform(descriptions)
    #lsa the data
    vec_desc = lsa.fit_transform(vec_desc)
    
    return vec_desc

#def topsis_based_similarity():
    #comparing words based off their synonyms using wordnet
    #learned from https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/
 #   w1 = wordnet.synset('run.v.01') # v here denotes the tag verb 
  #  w2 = wordnet.synset('sprint.v.01') 
   # print(w1.wup_similarity(w2))

    #https://medium.com/parrot-prediction/dive-into-wordnet-with-nltk-b313c480e788
    #set each word to it's wordnet word
    #print(w1.path_similarity(w2))
    

def kmeans(vectors, mini_batch=False, true_k = 20, predict = True):
    #setup kmeans clustering
    #k-means is optimizing a non-convex objective function, it will likely end up in a local optimum. Several runs 
    #with independent random init might be necessary to get a good convergence.
    if mini_batch == False:
        k_means = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=0)
    else:
        k_means = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, 
                                  batch_size=1000, verbose=0)
   
    #fit the data
    if predict == True:
        cluster = k_means.fit_predict(vectors)
    else:
        cluster = k_means.fit(vectors)
    
    return cluster

#not super worth using when a lot of the descriptors use the same words
def top_cluster_terms(descriptions, vec, true_k = 20, mini_batch=False):
    #build the model
    k_means = kmeans(vec, mini_batch=mini_batch, true_k = true_k, predict = False)
    stop_words = stopwords.words('english')
    tfv = TfidfVectorizer(input='content', stop_words = stop_words, ngram_range = (1,1), 
                          max_df=0.5, min_df=2, use_idf=True)
    order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]
    
    tfv.fit_transform(descriptions)
    terms = tfv.get_feature_names()
    
    print("Top terms per cluster:")
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
    return

# create dataframe, set the column to work off of
df = pd.read_csv(r'C:\Users\fitzg\Documents\Python Scripts\NLP\wine_desc.csv')
column_name = 'description'

# nlp (clean + wsd/taging + lematize + add related words + tfidf)
df = word_clean(df, column_name)
df[column_name] = df[column_name].apply(lambda sen: process_sentence(sen, add_words = True, 
    use_syn = True, use_hyper = True, use_hypo = True, use_hol = True, use_part_mer = True, use_sub_mer = True))
desc = df[column_name].values
vec = tfidf_lsa(desc)

# kmeans on the tfidf data
cluster_list = kmeans(vec)
df['cluster_num'] = pd.Series(cluster_list)