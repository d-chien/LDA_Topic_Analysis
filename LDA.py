import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm.auto import tqdm
from time import time

df = pd.read_csv('movie_data.csv', encoding = 'utf-8')

# setup CountVec as bag of word, to input to LDA
count = CountVectorizer(stop_words='english', max_df=.1,max_features = 5000)
'''
stop_words: built-in english stop words, so we don't have to use tf-idf for stop words
max_df: max frequency a word appear in all docs to prevent uncatched stopwords
max_features: limit vocabularies in a topic
'''

X = count.fit_transform(df.review.values)

# setup lda
lda = LatentDirichletAllocation(n_components=10,random_state=38, learning_method='batch')

x_topics = lda.fit_transform(X)  # get possibility to identify as topic n for each review (array(review_count, topics_count))
print(lda.components_.shape)  # (topic, vocs)

n_top_words = 5
feature_names = count.get_feature_names()  # aquire words in bag of word
for idx, topic in enumerate(lda.components_):  # lda.components_ would return weight of each word in bag for each topic
    print(f'Topic {idx}')
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))  # argsort gives the order index of each word sorted from min to max, so select last index would be the most important words of each topic

'''
validate
Topic 9
horror sex gore fans scary
'''

horror = x_topics[:,9].argsort()[::-1]  # x_topics[:,n] would be the possibility that the doc belongs to topic n. use argsort+reverse to get the docs with highest possibility.
for idx, moview_idx in enumerate(horror[:3]):
    print(f'Horror movie #{idx}')
    print(df.review[moview_idx][:300],'...')
