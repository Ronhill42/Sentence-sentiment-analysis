from dataclasses import dataclass
from typing import List, Literal
import fitz
from fitz import Page
import pandas as pd
from nltk import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
import re

"""
Use sentence_lists_by_sentiment(ID) to get 'positive', 'negative' and 'other' sentences from pdf with id = ID.

Use highlight_pdf(filepath, ID) to call sentence_lists_by_sentiment(ID) and highlight those sentences in the pdf.
"""


def fit(id_no) -> str:
    """
    Opens PDF file and gets all the text, joins strings, arranges full stops.
    Removes non aciis such as bullet points etc.
    
    Parameters:
        id_no (int): ID number of the PDF document, saved as './{id_no}.pdf'.
        
    Return:
        A string of all the text.
    """
    doc = fitz.open(f'./{id_no}.pdf')
    text = [page.get_text() for page in doc] 
    #Merge strings into one and add to list - full stops are added in certain locations to prompt the sentence splitter to divide strings.
    results = '. '.join(text)
    myres = []
    myres.append(str(results))
    #Regex Remove hyphens acting as bullet points
    res = [re.sub(r'[\W]*-[\W]', ". ", x) for x in myres]
    #re.sub[(r'\n-', ". ", x) for x in myres]
    #myregex = r'[\W]*'+ (chr(8212)) + '[\W]' 
    #rest = [re.sub(myregex, ". ", x) for x in res]

    #Text manipulation
    char = '\n'
    results = list(map(lambda x: x.replace(char, ''), res))
    #Remove duplicated full stops
    res = list(map(lambda x: x.replace('..', '.'), results))
    #Remove non ascii characters.
    ascii_ids = [*range(1,150), 8212, 8217]
    asciis = []
    for j in res:
        each_ascii = [i for i in j if ord(i) not in ascii_ids]
        asciis.append(each_ascii) 
    my_list = sum(asciis, [])
    #Remove duplicates
    my_list = sorted(set(my_list), key = my_list.index)
    result = []
    for j in res:
        for i in my_list:
            j = j.replace(i, '. ')           
        result.append(j)
    res = list(map(lambda x: x.replace('..', '.'), result)) 
    result = list(map(lambda x: x.replace(';', '.'), res)) 
    res = list(map(lambda x: x.replace(':', '.'), result)) 
    result = list(map(lambda x: x.replace('. .', '.'), res)) 
    res = list(map(lambda x: x.replace('.  ', '. '), result)) 
    return res

def sentence_list(text: str, max_size: int = 512) -> List[str]: 
    """
    Tokenises and splits text into sentences.
    
    Parameters:
        text (str): A single string of text.

    Returns:
        List of sentences.   
        """
    return [each for each in sent_tokenize(text) if len(each)<512]


def dataframe(id_no: int) -> pd.DataFrame:
    """
    Reads the pdf, converts the text into sentences, 
    places each sentence on an independent row in a dataframe, 
    manipulates sentences to be suitable inputs for the predicter function.
     
    Parameters:
        id_no (int): The ID number of the pdf.
         
    Returns:
        Dataframe containing all sentences of a pdf split into rows.
    """
    df = pd.DataFrame()
    mylist = fit(id_no)
    df["text"] = mylist # table_to_sentences(id_no) 
    df["id"] = f'{id_no}'
    df["sentence_column"] = df["text"].apply(lambda mylist: sentence_list(mylist))  
    df = df.explode("sentence_column")
    df.dropna(inplace=True)
    df.reset_index(inplace=True)  
    #Text management
    df["sentence_column"] = df["sentence_column"].map(lambda x: str(x).replace('e.g.',''))
    df["sentence_column"] = df["sentence_column"].map(lambda x: str(x).replace('.',''))
    df['sentence_column'] = df['sentence_column'].apply(lambda x: x.strip())
    df = df.drop_duplicates(keep='first')
    df = df[["id", "sentence_column"]]
    #Remove rows with empty text cell
    df = df[df['sentence_column'].astype(bool)] #Empty columns removed since False
    df = df.drop_duplicates(keep='first')
    df.reset_index(drop=True, inplace =True)
    return df

def predicter(x: int) -> pd.DataFrame:
    """
    Takes text from dataframe and passes through the finbert sentiment predicter.
    
    Parameters: 
        x (int): ID number
    
    Returns:
        Dataframe with sentences and their respective scores for Positive, Negative and Neutral sentiment.
    """
    df = dataframe(x)
    model = AutoModelForSequenceClassification.from_pretrained("finbert_trained2", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    pipe = pipeline("text-classification", model=AutoModelForSequenceClassification.from_pretrained("finbert_trained2"), top_k=None,
                    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert"), device=0)
    predictions = pipe(df[df.columns[1]].values.tolist()) #returns dictionaries containing label and score
    df.reset_index(drop=True, inplace=True)
    df["predictions"] = predictions 
    df = df.join(pd.json_normalize(df["predictions"].apply(lambda x: {i['label']: i['score'] for i in x})))
    del df["predictions"]
    return df

def overall_sentiment(x: int) -> pd.DataFrame:
    """
    Filters each sentence into a single primary sentiment depending on their scores; either Positive, Negative or Other.
    Note that for a sentence to be deemed as positive or negative, it's score for that sentiment must be greater than 
    the threshold value, otherwise it will be labelled as 'Other'.
    
    Parameters:
        x (int): ID number
    
    Returns:
        Dataframe with each sentence and the respective overall sentiment. No score.
    """
    df = predicter(x)
    conditions = [
        (df['positive']>0.9),
        (df['negative']>0.95),
        (df['positive']<0.9) & (df['negative']<0.95)
    ]
    values = ['Positive', 'Negative', 'Other']
    #Assigns sentence to one sentiment
    df['Sentiment']= np.select(conditions,values)
    del (df['negative'], df['positive'], df['neutral'])
    return df

def sentence_lists_by_sentiment(x: int) -> (List[str], List[str], List[str]):
    """
    Categorises sentences by sentiment.
    
    Parameters:
            x (int): ID number
            
    Returns:
        A list of sentences that are positive.
        A list of sentences that are negative.
        A list of sentences that are neither positive nor negative.
    """
    df = overall_sentiment(x)
    df_pos = df.loc[df['Sentiment'] == 'Positive']
    df_neg = df.loc[df['Sentiment'] == 'Negative']
    df_other = df.loc[df['Sentiment'] == 'Other']
    df_pos = df_pos['sentence_column'].values.tolist()
    df_neg = df_neg['sentence_column'].values.tolist()
    df_other = df_other['sentence_column'].values.tolist()
    return df_pos, df_neg, df_other

@dataclass
class WordAndColour:
    words: List[str]
    colour: List[float]
    type: Literal["positive", "negative", "other"]

def highlight_pdf(file: str, id: int):
    """
    Assigns a colour for each sentiment; green for positive, red for negative, blue for other.
    Iterates through document strings and uses highlight_string to annotate the words.

    Parameters:
        file (str): Filepath of pdf
        id (int): ID number

    Returns:
        Highlighted pdf document as a new file.
    """
    pos_words = sentence_lists_by_sentiment(id)[0]
    neg_words = sentence_lists_by_sentiment(id)[1]
    other_words = sentence_lists_by_sentiment(id)[2]
    colour_dict = {
        "positive": [0.65, 0, 0.65, 0],
        "negative": [0, 0.76, 0.74, 0],
        "other": [0.65, 0.14, 0, 0],
    }
    words_and_colours = [
        WordAndColour(
            words=pos_words, colour=colour_dict["positive"], type="positive"
        ),
        WordAndColour(
            words=neg_words, colour=colour_dict["negative"], type="negative"
        ),
        WordAndColour(
            words=other_words, colour=colour_dict["other"], type="other"
        ),
    ]
    doc = fitz.open(file)
    for i, page in enumerate(doc):
        for wt in words_and_colours:
            for w in wt.words:
                found = highlight_string(page, string=w, color=wt.colour) if len(w) > 1 else None
    doc.save(f'./{id}_highlighted.pdf')
    doc.close()
   
def highlight_string(page: Page, string: str, color: List[float]) -> bool:
    """
    Highlights string. 
    
    Parameters:
        page: Page 
        string: (str)
        color: (List[float])

    Returns:
        Boolean
    """
    found = False
    boxes = page.search_for(string, quads = True)
    if color is not None and boxes:
        for b in boxes:
            highlight = page.add_highlight_annot(b)
            highlight.set_colors({"stroke": color})
            highlight.update()
    return found

#highlight_pdf('./68269.pdf', 68269)
