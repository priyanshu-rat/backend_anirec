# app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
import os

from recommender import recommend, _ensure_loaded

app = FastAPI(title="Anime Recommender API")

# Allow cross-origin requests (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ensure model/data loaded on startup
@app.on_event("startup")
def startup_event():
    _ensure_loaded()

# ---- Routes ----
@app.get("/", tags=["ui"])
def top_anime(limit: int = 60):
    """
    Return top 'limit' anime sorted by score (for home page).
    """
    from recommender import df
    top = df.sort_values(by="score", ascending=False).head(limit)
    # Replace NaN with None for safe JSON encoding
    payload = top.replace({np.nan: None}).to_dict(orient="records")
    return JSONResponse(content=jsonable_encoder(payload))

@app.get("/search", tags=["search"])
def search(q: str = Query(..., min_length=2), top_each: int = 10, total_top: int = 40):
    """
    Search by query: returns main match (if exact or fuzzy) and recommendations.
    """
    recs = recommend(q, top_each=top_each, total_top=total_top)

    # Replace invalid float values (NaN, inf)
    clean_recs = []
    for rec in recs:
        clean_rec = {}
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean_rec[k] = None
            else:
                clean_rec[k] = v
        clean_recs.append(clean_rec)

    return JSONResponse(content=jsonable_encoder({"query": q, "recommendations": clean_recs}))

@app.get("/anime/{mal_id}", tags=["anime"])
def anime_detail(mal_id: int):
    """
    Return anime details for a given MAL id and recommendations for it.
    """
    from recommender import df
    anime = df[df["mal_id"] == mal_id]
    if anime.empty:
        raise HTTPException(status_code=404, detail="Anime not found")

    # Replace NaN with None for safe JSON
    record = anime.iloc[0].replace({np.nan: None}).to_dict()

    # Generate recommendations for this anime
    recs = recommend(record.get("title") or record.get("title_english") or "", total_top=20)

    clean_recs = []
    for rec in recs:
        clean_rec = {}
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean_rec[k] = None
            else:
                clean_rec[k] = v
        clean_recs.append(clean_rec)

    payload = {"anime": record, "recommendations": clean_recs}
    return JSONResponse(content=jsonable_encoder(payload))
