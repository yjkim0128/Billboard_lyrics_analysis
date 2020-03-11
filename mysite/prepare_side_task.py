import main_task as mt
import gensim
import pandas as pd
import numpy as np

def get_artist_dict(data, artists):
    lyrics_split = data['filtered_lyrics'].apply(lambda x: x.split())
    vocab_index = mt.get_vocab_index(lyrics_split)
    tfidf = mt.get_tfidf(lyrics_split, vocab_index)
    model = gensim.models.Word2Vec.load('billboard.model')
    w2v = mt.get_mean_w2v(lyrics_split, model)
    sentiment = data['polarity']
    artist_dict = {}
    for artist in artists:
        artist_i = list(data[data['artist'] == artist].index)
        artist_tfidf = np.mean(tfidf[artist_i], axis=0)
        artist_w2v = np.mean(w2v[artist_i], axis=0)
        artist_sentiment = np.mean(sentiment[artist_i])
        artist_dict[artist] = (artist_tfidf, artist_w2v, artist_sentiment)
    return artist_dict

def find_most_similar(name, artists, artist_dict):
    n = len(artists)
    distance = float('inf')
    most_similar = None
    for artist in artists:
        if name != artist:
            d_tfidf = 1 - np.dot(artist_dict[name][0], artist_dict[artist][0])
            d_w2v = 1 - np.dot(artist_dict[name][1], artist_dict[artist][1])
            d_sentiment = abs(artist_dict[name][2] - artist_dict[artist][2])
            d = d_tfidf * 10 ** 3 + 2 * d_w2v + 2 * d_sentiment * 10 ** -1
            if d < distance:
                distance = d
                most_similar = artist

    return most_similar

def create_most_similar():
    data = mt.data
    artists = list(data['artist'].unique())
    artist_dict = get_artist_dict(data, artists)
    
    most_similar = []
    for artist in artists:
        closest_artist = find_most_similar(artist, artists, artist_dict)
        most_similar.append(closest_artist)
    df = pd.DataFrame([most_similar])
    df.columns = artists
    return df

most_similar_df = create_most_similar()
most_similar_df.to_csv('most_similar.csv', index=False, sep='|')



    
    
