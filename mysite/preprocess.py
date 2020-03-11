### the functions in this file are used to save the data beforehand
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import gensim

data = pd.read_csv('artist_lyrics.txt', sep='|')
sid = SentimentIntensityAnalyzer()

def filter_lyrics(lyric):
    words_tokens = lyric.split()
    filtered_words = []
    for word in words_tokens:
        word = word.lower().strip(string.punctuation)
        filtered_words.append(word)
    return ' '.join(filtered_words)

def create_filtered_lyrics_col(df):
    df['filtered_lyrics'] = df['lyric'].apply(filter_lyrics)

def find_sentiment(lyric):
    return sid.polarity_scores(lyric)['compound']

def create_sentiment_col(df):
    df['polarity'] = df['filtered_lyrics'].apply(find_sentiment)

def create_decade_col(df):
    df['decade'] = None
    decades = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
    for decade in decades:
        df_bool = (decade <= data['year']) & (data['year'] < decade + 10)
        df.loc[df_bool, 'decade'] = str(decade)

def create_model(df):
    lyrics_split = data['filtered_lyrics'].apply(lambda x: x.split())
    model = gensim.models.Word2Vec(sentences=lyrics_split, size=100, window=5)
    return model


create_filtered_lyrics_col(data)
create_sentiment_col(data)
create_decade_col(data)
data.to_csv('processed_lyrics.csv', index=False, sep='|')
model = create_model(data)
model.save('billboard.model')


