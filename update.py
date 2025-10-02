import pickle, os, secure_pickle

CACHE = "backend/data/known_faces.pkl"
with open(CACHE, "rb") as f:
    c = pickle.load(f)

new_files, new_embs, new_meta = [], [], {}
for fn, emb in zip(c["files"], c["embeddings"]):
    uid = fn.split(".")[0]            # strip at first dot
    new_files.append(uid)
    new_embs.append(emb)
    new_meta[uid] = c["metadata"][fn]

c.update(files=new_files, embeddings=new_embs, metadata=new_meta)
with open(CACHE, "wb") as f:
    pickle.dump(c, f)
print("✓ cache normalised to bare UIDs")
