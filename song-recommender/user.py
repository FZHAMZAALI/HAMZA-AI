"""Personalized recommender: user ki pasand yaad rakhta hai.

    python user.py like    hamza "Tum Hi Ho"        # song pasand aaya
    python user.py dislike hamza "Baby"             # pasand nahi aaya
    python user.py next    hamza "Humdard"          # is bajte song + hamari pasand ke hisaab se agle 10
    python user.py next    hamza                    # sirf pasand ke hisaab se
    python user.py show    hamza                    # profile dekho

Taste = pasand kiye songs ka average point, minus thoda dislike wale songs ki taraf se.
"""
import json, os, sys
import numpy as np
from recommend import df, X, knn, find

NOW_W, LIKE_W, DISLIKE_W = 0.5, 0.5, 0.3     # kisko kitna weight (badal kar dekho)


def load(user):
    f = f"profile_{user}.json"
    return json.load(open(f)) if os.path.exists(f) else {"likes": [], "dislikes": []}


def save(user, p):
    json.dump(p, open(f"profile_{user}.json", "w"))


def label(i):
    return f"{df.track_name[i]} - {df.artists[i]}"


def recommend_for(user, now=None, n=10):
    p = load(user)
    likes, dislikes = p["likes"], p["dislikes"]
    parts, weights = [], []
    if now is not None:
        parts.append(X[now]); weights.append(NOW_W)
    if likes:
        parts.append(X[likes].mean(axis=0)); weights.append(LIKE_W)
    if not parts:
        return []
    q = sum(w * v for w, v in zip(weights, parts)) / sum(weights)
    if dislikes:
        q = q - DISLIKE_W * (X[dislikes].mean(axis=0) - q)   # dislike se door dhakelo
    _, idx = knn.kneighbors(q.reshape(1, -1), n_neighbors=60)
    skip = set(likes) | set(dislikes) | ({now} if now is not None else set())
    skip_names = {df.track_name[i].lower() for i in skip}
    out = []
    for j in idx[0]:
        name = df.track_name[j].lower()
        if j in skip or name in skip_names:
            continue
        skip_names.add(name)
        out.append(j)
        if len(out) == n:
            break
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, user = sys.argv[1], sys.argv[2]
    song = " ".join(sys.argv[3:])
    p = load(user)
    if cmd in ("like", "dislike"):
        i = find(song)
        if i is None:
            sys.exit("Song nahi mila.")
        i = int(i)
        for k in ("likes", "dislikes"):
            if i in p[k]:
                p[k].remove(i)
        p[cmd + "s"].append(i)
        save(user, p)
        print(f"{cmd}: {label(i)}")
    elif cmd == "show":
        print("Likes   :", [label(i) for i in p["likes"]])
        print("Dislikes:", [label(i) for i in p["dislikes"]])
    elif cmd == "next":
        now = find(song) if song else None
        if song and now is None:
            sys.exit("Song nahi mila.")
        recs = recommend_for(user, None if now is None else int(now))
        if not recs:
            sys.exit("Pehle kuch songs like karo.")
        print(f"\n{user} ke liye agle 10 songs:")
        for k, j in enumerate(recs, 1):
            print(f"{k:2}. {label(j)}")
