import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
from sentence_transformers import SentenceTransformer

# ---------- Configuration ----------
CSV_PATH = os.path.join(os.path.dirname(__file__), "finalAnime.csv")
EMB_PATH = os.path.join(os.path.dirname(__file__), "anime_bert_embeddings_cpu.pt")
NORM_PATH = os.path.join(os.path.dirname(__file__), "anime_normalized_embeddings.npy")
MODEL_NAME = "paraphrase-MiniLM-L6-v2"
MODEL_CACHE_DIR = os.path.join("/tmp", "anime_recommender_model")

# ---------- Globals ----------
df = None
embeddings = None
norm_embeddings = None
model = None
all_genres = None

# ---------- Utilities ----------
def _normalize(series):
    s = np.array(series, dtype=np.float64)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if mx - mn < 1e-8:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)

def _ensure_loaded():
    """Load dataset, embeddings, and model once (with cached normalized embeddings)."""
    global df, embeddings, norm_embeddings, model, all_genres

    if df is not None and norm_embeddings is not None and model is not None:
        return

    print("📦 Loading dataset and embeddings...")
    df_local = pd.read_csv(CSV_PATH)
    df_local = df_local.reset_index(drop=True)

    # Ensure required columns exist
    for col in ["title", "title_english", "genres", "image_url", "score", "mal_id", "synopsis"]:
        if col not in df_local.columns:
            df_local[col] = ""

    # Prepare clean titles
    df_local["clean_title"] = df_local["title"].fillna("").astype(str).str.lower().str.strip()
    df_local["clean_eng_title"] = df_local["title_english"].fillna("").astype(str).str.lower().str.strip()
    df_local["search_title"] = df_local["clean_eng_title"].where(df_local["clean_eng_title"] != "", df_local["clean_title"])

    # Load embeddings
    emb = torch.load(EMB_PATH, map_location=torch.device("cpu"))
    if isinstance(emb, torch.Tensor):
        emb_arr = emb.cpu().numpy()
    elif isinstance(emb, (list, tuple)):
        emb_arr = np.stack([e.cpu().numpy() if isinstance(e, torch.Tensor) else np.array(e) for e in emb])
    else:
        emb_arr = np.array(emb)

    # Normalized embeddings cache
    if os.path.exists(NORM_PATH):
        norm_emb = np.load(NORM_PATH)
        print("✅ Loaded normalized embeddings from cache.")
    else:
        print("⚙️ Computing normalized embeddings (first time)...")
        norm_emb = emb_arr / np.linalg.norm(emb_arr, axis=1, keepdims=True)
        np.save(NORM_PATH, norm_emb)
        print("✅ Saved normalized embeddings to cache.")

    # Load model
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    model_local = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)

    # Build genre set
    genres_set = set()
    for g in df_local["genres"].dropna().astype(str):
        for item in [x.strip().lower() for x in g.split(",")]:
            if item:
                genres_set.add(item)

    df, embeddings, norm_embeddings, model, all_genres = df_local, emb_arr, norm_emb, model_local, genres_set

def _to_serializable(record):
    out = {}
    for k, v in record.items():
        if isinstance(v, (np.generic, np.ndarray)):
            out[k] = v.item() if np.ndim(v) == 0 else v.tolist()
        else:
            out[k] = v if (v is None or not (isinstance(v, float) and np.isnan(v))) else None
    return out

# ---------- Recommendation Function ----------
def recommend(query: str,
              top_each: int = 10,
              total_top: int = 40,
              w_sim: float = 0.6,
              w_score: float = 0.4,
              fuzzy_cutoff: int = 80):
    """Main hybrid recommender. Returns list of JSON-serializable recommendations."""
    _ensure_loaded()

    q = str(query).strip().lower()
    results_frames = []
    used_idx = set()
    idx = None
    genre_found = False

    # 1️⃣ Exact Match
    mask_exact = (df["clean_title"] == q) | (df["clean_eng_title"] == q)
    exact_matches = df[mask_exact]
    if not exact_matches.empty:
        idx = int(exact_matches.index[0])
        print(f"🎯 Exact match found: {df.loc[idx, 'title']}")

        # Semantic search
        query_vec = norm_embeddings[idx]
        sims = np.dot(norm_embeddings, query_vec)
        sim_idx = np.argsort(sims)[::-1][1:top_each+1]
        sem_recs = df.iloc[sim_idx].copy()
        sem_recs["similarity"] = sims[sim_idx]
        sem_recs["norm_score"] = _normalize(sem_recs["score"].fillna(0))
        sem_recs["combined_score"] = w_sim * sem_recs["similarity"] + w_score * sem_recs["norm_score"]
        results_frames.append(sem_recs)
        used_idx.update(sem_recs.index.tolist())

        # Genre expansion
        target_genres = df.loc[idx, "genres"]
        target_set = set([g.strip().lower() for g in str(target_genres).split(",") if g.strip()])
        if target_set:
            def shares_genre(cell):
                return len(target_set.intersection(set([x.strip().lower() for x in str(cell).split(",")]))) > 0
            genre_candidates = df[df["genres"].apply(shares_genre)]
            genre_candidates = genre_candidates[~genre_candidates.index.isin(used_idx)]
            genre_candidates["norm_score"] = _normalize(genre_candidates["score"].fillna(0))
            genre_candidates["combined_score"] = w_score * genre_candidates["norm_score"]
            top_genre = genre_candidates.sort_values("combined_score", ascending=False).head(top_each)
            results_frames.append(top_genre)
            used_idx.update(top_genre.index.tolist())

        genre_found = True

    # 2️⃣ Genre Query (if not exact)
    if not genre_found and q in all_genres:
        print(f"🎨 Genre match found: {q}")
        def contains_genre(cell): return q in str(cell).lower()
        genre_candidates = df[df["genres"].apply(contains_genre)].copy()
        genre_candidates["norm_score"] = _normalize(genre_candidates["score"].fillna(0))
        genre_candidates["combined_score"] = genre_candidates["norm_score"]
        top_genre = genre_candidates.sort_values("combined_score", ascending=False).head(total_top)
        return [_to_serializable(r) for r in top_genre.to_dict(orient="records")]

    # 3️⃣ Semantic fallback (query text)
    q_emb = model.encode(q, convert_to_tensor=False, normalize_embeddings=True)
    q_emb = np.array(q_emb).reshape(1, -1)
    cos_q = cosine_similarity(q_emb, norm_embeddings)[0]
    sem_idx = np.argsort(cos_q)[::-1][:top_each]
    sem_recs = df.iloc[sem_idx].copy()
    sem_recs["similarity"] = cos_q[sem_idx]
    sem_recs["norm_score"] = _normalize(sem_recs["score"].fillna(0))
    sem_recs["combined_score"] = w_sim * sem_recs["similarity"] + w_score * sem_recs["norm_score"]
    results_frames.append(sem_recs)
    used_idx.update(sem_idx)

    # 4️⃣ Fuzzy fallback
    best = process.extractOne(q, df["search_title"].tolist(), scorer=fuzz.token_sort_ratio)
    if best and best[1] >= fuzzy_cutoff:
        match_title = best[0]
        fidx = df[df["search_title"] == match_title].index[0]
        print(f"🔍 Fuzzy match → {match_title} ({best[1]}%)")

        f_vec = norm_embeddings[fidx]
        sims_f = np.dot(norm_embeddings, f_vec)
        sim_idx = np.argsort(sims_f)[::-1][1:top_each+1]
        fuzzy_recs = df.iloc[sim_idx].copy()
        fuzzy_recs["similarity"] = sims_f[sim_idx]
        fuzzy_recs["norm_score"] = _normalize(fuzzy_recs["score"].fillna(0))
        fuzzy_recs["combined_score"] = 0.6 * w_sim * fuzzy_recs["similarity"] + 0.4 * w_score * fuzzy_recs["norm_score"]
        results_frames.append(fuzzy_recs)

    # 5️⃣ Combine all results
    if results_frames:
        combined = pd.concat(results_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="title").reset_index(drop=True)
        if "combined_score" not in combined.columns:
            combined["combined_score"] = 0.0
        combined = combined.sort_values("combined_score", ascending=False).head(total_top)
        return [_to_serializable(r) for r in combined.to_dict(orient="records")]

    return []
