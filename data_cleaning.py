import pandas as pd
import re
from tqdm.auto import tqdm

tqdm.pandas()

df = pd.read_csv('movie_data.csv',encoding = 'utf-8')

def cleaning(text):
    trans_text = re.sub('<[^>]*>','',text)  # starts with "<", ends with ">", contains undefined number (*) of "not >" characters(^>)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)  # might starts with one of ":/;/=", ends with
    return_text = re.sub('[\W]+',' ',trans_text.lower())+ ' '.join(emoticons).replace('-','')  # change non alphabet/digit characters to space and add back emoticons
    return return_text

df.review = df.review.progress_apply(cleaning)
df.to_csv('movie_data.csv', encoding = 'utf-8')