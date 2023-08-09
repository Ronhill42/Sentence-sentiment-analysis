from transformers import pipeline
import pandas as pd
import re
from functions import split_into_sentences
import numpy as np
from typing import List
from nltk import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

#Figure out how I filtered by management letter

#Sentence splitter
def sentence_list(text: str, max_size: int = 512) -> List[str]: 
    x=[]
    for each in sent_tokenize(text):
        x.append(each if len(each)<512 else "")
    return x
   #ignores sentences 512 characters or longer
 #return [each for each in sent_tokenize(text) if len(each)<512]
  
#Taking number of management letters you want to use
def get_corpus() -> pd.DataFrame : 
    #Unpickle File
    with open("./sentence_sentiment_analysis/corpus_training_data.pkl", "rb") as file:
        data = pd.read_pickle(file)
    data.reset_index(inplace=True)
    data["sentence_column"] = data["text"].apply(lambda text: sentence_list(text))  
    
    data_final = data.explode("sentence_column")
    
    return data_final[["index", "sentence_column"]].head(20)


    
def just_sentences(): 
    df = get_corpus()
    return df[df.columns[1]].values.tolist()



def predicter() -> List[dict[str, str]]:
    #Run sentences through finbert
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    pipe = pipeline("text-classification", model=AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert"), top_k=None,
                    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert"), device=0)
    predictions = pipe(just_sentences()) #returns dictionaries containing label and score
    df = get_corpus()
    df["predictions"] = predictions 
    df = df.join(pd.json_normalize(df["predictions"].apply(lambda x: {i['label']: i['score'] for i in x})))

    #Split sentiments into their own columns
    del df["predictions"]
    return df

print(predicter())
"""
def split_scores() -> pd.DataFrame:
    neutral_scores=[]
    negative_scores = []
    positive_scores = []
    first_key: str = 'label'
    for each in predicter():
        for j in each:
            first_value = j[first_key]
            if first_value == 'neutral':
                neutral_scores.append([j['score']])
            if first_value == 'negative':
                negative_scores.append([j['score']])
            if first_value == 'positive':
                positive_scores.append([j['score']])
    df = get_corpus()
    df['Neutral'] = neutral_scores
    df['Negative']= negative_scores
    df['Positive']= positive_scores
    return df[["index", "sentence_column", "Neutral", "Negative", "Positive"]]
"""


"""
#Troubleshooting - check what text looks like in text_test_first_line file
def file_check():
    lines = '\n'.join([i for i in just_sentences])
    text_files = open("text_test_first_line", "w")
    n = text_files.write(lines) #Transform dataframe to a string
    text_files.close()


#file_check()
"""