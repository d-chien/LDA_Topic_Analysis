import tarfile
with tarfile.open('./ACL IMDB v1 Sentiment Analysis.tar.gz','r:gz') as tar:
    tar.extractall()