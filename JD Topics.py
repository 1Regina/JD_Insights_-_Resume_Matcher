#%% 
# Import libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import gensim
from gensim import corpora, models, similarities, matutils
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize,RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import time

#%%
# PREPROCESSING
# Import Dataset and remove empty rows
df_all = pd.read_csv(r"C:\Users\regina\Desktop\Metis\Metis Projects\JD Resume Matcher\US JD data.csv")
df_all_positions = df_all[pd.notnull(df_all['position'])]
#df_all_positions.to_csv("not_null.csv", index = False)
df_all_positions.head()
df_all_positions.shape

#%% 
#Wordcloud of job_titles
df_all = pd.read_csv(r"C:\Users\regina\Desktop\Metis\Metis Projects\JD Resume Matcher\US JD data.csv")
%config InlineBackend.figure_formats = ['retina']     # sets backend to render higher res images
wordcloud = WordCloud(background_color='white',colormap = "viridis", max_font_size = 50).generate_from_frequencies(df_all['position'].value_counts())
# wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('titles.png',format = 'png', dpi=1000)
plt.savefig('titles.svg',format = 'svg', dpi=1000)
plt.show()
#%%
# Filter for positions containing "Data Scientist"
df_data_scientist  = df_all_positions.set_index('position').filter(like='Data Scien', axis=0)
df_data_scientist.shape
df_data_scientist.head()
df_data_scientist.to_csv("Data Scien.csv")

# Remove punctuation 
df_data_scientist['no_punctuation'] = df_data_scientist['description'].str.replace('[^\w\s]','')

# Change to lower case
df_data_scientist ['lower_case'] = df_data_scientist['no_punctuation'].astype(str).str.lower()
df_data_scientist.head()

# Apply word tokenizer
df_data_scientist['tokenized_text'] = df_data_scientist['lower_case'].apply(word_tokenize)
# df_data_scientist.to_csv("df_tokenized_text.csv", index=True)
df_data_scientist.head()

# Remove stop words
df_data_scientist['key_words'] =df_data_scientist['tokenized_text'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])
#df_data_scientist.to_csv("df_stop_words.csv", index=True)
df_data_scientist.head()

# Word stemming
porter_stemmer = PorterStemmer()
df_data_scientist['stem']=df_data_scientist['key_words'].apply(lambda x : [porter_stemmer.stem(y) for y in x])
df_data_scientist.to_csv("df_stem.csv", index=True)
df_data_scientist.head()

#%%
df_data_scientist = pd.read_csv(r"C:\Users\regina\Desktop\Metis\Metis Projects\JD Resume Matcher\Data Scien.csv")
#Wordcloud of companies
%config InlineBackend.figure_formats = ['retina']     # sets backend to render higher res images
wordcloud = WordCloud(background_color='white',colormap = "inferno", max_font_size = 50).generate_from_frequencies(df_data_scientist['company'].value_counts())
# wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('companies.png',format = 'png', dpi=1000)
plt.savefig('companies.svg',format = 'svg', dpi=1000)
plt.show()

#Wordcloud of location
%config InlineBackend.figure_formats = ['retina']     # sets backend to render higher res images
wordcloud = WordCloud(background_color='white',colormap = "magma", max_font_size = 50).generate_from_frequencies(df_data_scientist['location'].value_counts())
# wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('location.png',format = 'png', dpi=1000)
plt.savefig('location.svg',format = 'svg', dpi=1000)
plt.show()

#%%
# COUNT VECTORIZER
# Join the tokenized words for count vectorizing
df_data_scientist['joined_Sent'] = [' '.join(map(str, indStem)) for indStem in df_data_scientist['stem']]
df_data_scientist.head()

# Document-Term Matrix (Count Vectorizer)
word_vectorizer = CountVectorizer(ngram_range=(1,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(df_data_scientist["joined_Sent"])
df_doc_term = pd.DataFrame(sparse_matrix.toarray(), columns=word_vectorizer.get_feature_names())
df_doc_term.shape
df_doc_term.tail()
df_doc_term.to_csv("df_doc_term.csv", index=True)

df_doc_term.tail()
#%%
# TOPIC MODELLING - COUNT VECTORIZER
# 1. LDA , Genism
time1 = time.time()
# Convert sparse matrix of counts to a gensim corpus 
corpus = matutils.Sparse2Corpus(sparse_matrix)
print (corpus)
# Map matrix rows to words (tokens)
id2word = dict((v, k) for k, v in word_vectorizer.vocabulary_.items())
len(id2word)
# Create lda model (equivalent to "fit" in sklearn)
lda = models.LdaModel(corpus=corpus, num_topics=3, id2word=id2word, passes=5)
# 10 most important words for each of the 3 topics
lda.print_topics()
# Transform the docs from word space to topic space
lda_corpus = lda[corpus]
# Store the doc topic vectors in a list for review
lda_docs = [doc for doc in lda_corpus]
# Find the document vectors in the topic space for the first 10 documents
lda_docs[0:10] 
time2 = time.time()
time_taken = time2 - time1
print(time_taken)

#%%
lda.print_topics()

#%%
# 2. LSA aka SVD
# Topic Modeling with Matrix Factorization and LSA (Latent Semantic Analysis) aka Singular Value Decomposition (SVD)
time1 = time.time()
lsa = TruncatedSVD(4)
# Transform the doc-term matrix to doc-topic matrix
df_doc_topic = lsa.fit_transform(df_doc_term)
lsa.explained_variance_ratio_
# Getting the U-matrix in Decomposition
topic_word = pd.DataFrame(lsa.components_.round(10), # 10 for dec pts
             index = ["component_1","component_2","component_3","component_4"],
             columns = word_vectorizer.get_feature_names()) # notes is vectoriser only 
topic_word
# Top 10 key words for each of the 6 topics
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lsa, word_vectorizer.get_feature_names(), 60)
time2 = time.time()
time_taken = time2 - time1
print(time_taken)

#%%
# 3. Non-Negative Matrix Factorization (NMF)
#Non-Negative Matrix Factorization (NMF)
time1 = time.time()
nmf_model = NMF(4)
doc_topic = nmf_model.fit_transform(df_doc_term)
topic_word = pd.DataFrame(nmf_model.components_.round(10),
             index = ["component_1","component_2","component_3","component_4"],
             columns = word_vectorizer.get_feature_names())
topic_word
display_topics(nmf_model, word_vectorizer.get_feature_names(), 60)
time2 = time.time()
time_taken = time2 - time1
print(time_taken)

#%%
# TOPIC MODELLING - TF-IDF VECTORIZER
# Document-Term Matrix - TF-IDF
cv_tfidf = TfidfVectorizer(ngram_range=(1,2), analyzer='word')
sparse_matrix = cv_tfidf.fit_transform(df_data_scientist["joined_Sent"])
df_doc_term_TFIDF = pd.DataFrame(sparse_matrix.toarray(), columns=cv_tfidf.get_feature_names())
df_doc_term_TFIDF.shape
df_doc_term_TFIDF.tail()
# df_doc_term_TFIDF.to_csv("df_doc_term_TFIDF.csv", index=True)

#%%
# 1. LDA , Genism
lda.print_topics()
time1 = time.time()
# Convert sparse matrix of counts to a gensim corpus 
corpus = matutils.Sparse2Corpus(sparse_matrix)
# Map matrix rows to words (tokens)
id2word = dict((v, k) for k, v in cv_tfidf.vocabulary_.items())
# len(id2word)
# Create lda model (equivalent to "fit" in sklearn)
lda = models.LdaModel(corpus=corpus, num_topics=3, id2word=id2word, passes=5)
# 10 most important words for each of the 3 topics
lda.print_topics()
# Transform the docs from word space to topic space
lda_corpus = lda[corpus]
# Store the doc topic vectors in a list for review
lda_docs = [doc for doc in lda_corpus]
# Find the document vectors in the topic space for the first 10 documents
lda_docs[0:50] 
time2 = time.time()
time_taken = time2 - time1
print(time_taken)

#%%
# 2. LSA aka SVD
# Topic Modeling with Matrix Factorization and LSA (Latent Semantic Analysis) aka Singular Value Decomposition (SVD)
time1 = time.time()
lsa = TruncatedSVD(4)
# Transform the doc-term matrix to doc-topic matrix
df_doc_topic = lsa.fit_transform(df_doc_term_TFIDF)
lsa.explained_variance_ratio_
# Getting the U-matrix in Decomposition
topic_word = pd.DataFrame(lsa.components_.round(10), # 10 for dec pts
             index = ["component_1","component_2","component_3", "component_4"],
             columns = cv_tfidf.get_feature_names()) # notes is vectoriser only 
topic_word
# Top 10 key words for each of the 6 topics
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lsa, cv_tfidf.get_feature_names(), 60)
time2 = time.time()
time_taken = time2 - time1
print(time_taken)

#%%
#3. Non-Negative Matrix Factorization (NMF)
#Non-Negative Matrix Factorization (NMF)
time1 = time.time()
nmf_model = NMF(4)
doc_topic = nmf_model.fit_transform(df_doc_term_TFIDF)
topic_word = pd.DataFrame(nmf_model.components_.round(10),
             index = ["component_1","component_2","component_3","component_4"],
             columns = cv_tfidf.get_feature_names())
topic_word
display_topics(nmf_model, cv_tfidf.get_feature_names(), 60)
time2 = time.time()
time_taken = time2 - time1
print(time_taken)

# #%%
# #word2Vec
# #1. Google word embedding
# google_vec_file = 'C:\\Users\\regina\\Desktop\\Metis\\SGP19_DS0\\curriculum\\project-04\\GoogleNews-vectors-negative300.bin'
# model = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)

# model['data']

# model.most_similar('data science',topn=20)
# model.most_similar(positive=['woman', 'king'], negative=['man'])

# #%%
# #2. Glove word embedding (Stanford)
# glove_file = glove_dir = 'C:\\Users\\regina\\Desktop\\Metis\\SGP19_DS0\\curriculum\\project-04\\glove.6B\\glove.6B.100d.txt'

# w2v_output_file = 'C:\\Users\\regina\\Desktop\\Metis\\SGP19_DS0\\curriculum\\project-04\\glove.6B\\glove.6B.100d.txt.w2v'

# # The following utility converts file formats
# gensim.scripts.glove2word2vec.glove2word2vec(glove_file, w2v_output_file)
# model = gensim.models.KeyedVectors.load_word2vec_format(w2v_output_file, binary=False)

# model.most_similar('data' ,topn=20)
# model.most_similar(positive=['woman', 'king'], negative=['man'])