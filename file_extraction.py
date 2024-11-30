import tarfile
import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm
import shutil
with tarfile.open('./ACL IMDB v1 Sentiment Analysis.tar.gz','r:gz') as tar:
    tar.extractall()

basepath = './aclImdb'
labels = {'pos':1, 'neg':0}

df = pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        my_path = os.path.join(basepath,s,l)
        for file in tqdm(sorted(os.listdir(my_path))):
            with open(os.path.join(my_path,file),'r',encoding = 'utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt,labels[l]]], ignore_index = True) # add label to comments to identify pos/neg comments

df.columns = ['review','sentiment']
np.random.seed(38)
df = df.reindex(np.random.permutation(df.index)) # rearrange order
df.to_csv('movie_data.csv',index = False, encoding = 'utf-8')

# remove unzipped data
shutil.rmtree(os.path.join(os.getcwd(),'aclImdb'))
print(f'upzipped files deleted')