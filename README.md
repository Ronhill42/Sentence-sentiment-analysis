# Sentence-sentiment-analysis
Testing and Training the ProsusAI/finbert NLP model

Main.py
This code uses the Finbert natural language processing model to predict the nature of passed text as a sentiment namely, 'positive', 'negative', or 'neutral'. 
The text has been taken from a dataframe with columns related to an id number and its respective list of sentences, split using a natural language tokeniser, 
before being passed through the prediction model.

Trainer.py
This code uses a dataset (which can be split into training and test data) as an input to the finbert training model. The model is trained and then saved as a entirely new model.


