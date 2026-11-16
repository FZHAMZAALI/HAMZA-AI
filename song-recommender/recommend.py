"""Song recommender: bajta hua song do, agle 10 milte-julte songs pao.

Idea: har song ke audio features (energy, mood, tempo...) ek "point" hain.
Jo songs us point ke sabse paas hain, woh sabse milte-julte hain (K-Nearest Neighbors).

    python recommend.py "Shape of You"
"""
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

df = pd.read_csv("songs.csv")
df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)
X = StandardScaler().fit_transform(df[FEATURES])      # sab features ko same scale par lao
# Genre bhi jodo (pop, hip-hop...), taaki milte-julte *type* ke songs aayein. GENRE_WEIGHT badhao to genre zyada matter karega.
GENRE_WEIGHT = 2.0
genres = pd.get_dummies(df.track_genre).to_numpy(dtype=float) * GENRE_WEIGHT
X = np.hstack([X, genres])
knn = NearestNeighbors(n_neighbors=30).fit(X)


def find(name):
    """Naam se song ka index dhoondo (sabse popular match). Na mile to None."""
    hits = df[df.track_name.str.lower() == name.lower()]
    if hits.empty:
        hits = df[df.track_name.str.lower().str.contains(name.lower(), regex=False)]
    return None if hits.empty else hits.sort_values("popularity", ascending=False).index[0]


def recommend(name, n=10):
    i = find(name)
    if i is None:
        return None, None
    _, idx = knn.kneighbors(X[[i]])
    seen, out = {df.track_name[i].lower()}, []
    for j in idx[0][1:]:
        key = df.track_name[j].lower()
        if key not in seen:                                         # same naam ke duplicate hatao
            seen.add(key)
            out.append(j)
        if len(out) == n:
            break
    return df.loc[i], df.loc[out]


if __name__ == "__main__":
    song, recs = recommend(" ".join(sys.argv[1:]) or "Shape of You")
    if song is None:
        sys.exit("Song nahi mila, koi aur naam try karo.")
    print(f"\nBajta hua: {song.track_name} - {song.artists}\n\nAgle 10 songs:")
    for k, (_, r) in enumerate(recs.iterrows(), 1):
        print(f"{k:2}. {r.track_name} - {r.artists}")
