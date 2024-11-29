import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
np.set_printoptions(precision = 2)

# setup vectorizer and doc
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'the sun is shining, the weather is sweet',
    'and one and one is two',
    'why we need to work for one third of our life'
])
bag = count.fit_transform(docs)

# count voc and no.
print(count.vocabulary_)
# print no and mapping freq
print(bag.toarray())

tfidf = TfidfTransformer(use_idf = True, norm = 'l2', smooth_idf = True)
print(tfidf.fit_transform(bag).toarray())



