"""Спільні функції Hook Finder (етап 0): метадані FMA, кеш ембедингів, пошук."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUDIO_DIR = DATA / "fma_small"
EMB_DIR = DATA / "embeddings"

CHUNK_S = 10.0  # довжина шматка, с
HOP_S = 5.0  # крок між шматками, с
METHODS = {"clap": "CLAP", "baseline": "Ручні ознаки"}


def track_path(track_id: int) -> Path:
    name = f"{int(track_id):06d}"
    return AUDIO_DIR / name[:3] / f"{name}.mp3"


def load_small_metadata() -> pd.DataFrame:
    """Таблиця треків fma_small: genre, artist, title, license."""
    t = pd.read_csv(DATA / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])
    s = t[t[("set", "subset")] == "small"]
    return pd.DataFrame(
        {
            "genre": s[("track", "genre_top")],
            "artist": s[("artist", "name")],
            "title": s[("track", "title")],
            "license": s[("track", "license")],
        }
    ).rename_axis("track_id")


def balanced_subset(meta: pd.DataFrame, subset_size: int, seed: int = 0) -> pd.DataFrame:
    """Порівну треків кожного жанру; subset_size >= len(meta) — усі треки."""
    if subset_size >= len(meta):
        return meta
    per_genre = subset_size // meta["genre"].nunique()
    parts = [g.sample(per_genre, random_state=seed) for _, g in meta.groupby("genre")]
    return pd.concat(parts).sort_index()


def load_cache(subset_size: int) -> dict:
    """Кеш, який створює build_embeddings.py. Нічого не перераховує."""
    tag = f"n{subset_size}"
    tracks = pd.read_csv(EMB_DIR / f"tracks_{tag}.csv", index_col="track_id")
    chunks = pd.read_csv(EMB_DIR / f"chunks_{tag}.csv")
    return {
        "tracks": tracks,
        "chunks": chunks,
        "clap": np.load(EMB_DIR / f"clap_{tag}.npy"),
        "baseline": np.load(EMB_DIR / f"baseline_{tag}.npy"),
    }


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def track_scores(emb: np.ndarray, chunks: pd.DataFrame, query_vecs: np.ndarray):
    """Косинусна подібність запитів до треків: max по шматках треку.

    emb — L2-нормовані вектори шматків (порядок як у chunks, шматки треку йдуть підряд).
    Повертає (track_ids, scores[n_queries, n_tracks], best_chunk[n_queries, n_tracks]).
    """
    tids = chunks["track_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tids[1:] != tids[:-1]])
    sims = np.atleast_2d(query_vecs) @ emb.T
    scores = np.maximum.reduceat(sims, starts, axis=1)
    best = np.empty_like(scores, dtype=int)
    ends = np.r_[starts[1:], len(tids)]
    for j, (a, b) in enumerate(zip(starts, ends)):
        best[:, j] = a + sims[:, a:b].argmax(axis=1)
    return tids[starts], scores, best


def top_k(emb: np.ndarray, chunks: pd.DataFrame, query_chunk: int, k: int = 5) -> pd.DataFrame:
    """Top-k треків для шматка-запиту, без треку самого запиту."""
    tids, scores, best = track_scores(emb, chunks, emb[query_chunk])
    scores, best = scores[0].copy(), best[0]
    scores[tids == chunks["track_id"].iloc[query_chunk]] = -np.inf
    order = np.argsort(-scores)[:k]
    return pd.DataFrame(
        {
            "track_id": tids[order],
            "score": scores[order],
            "chunk": best[order],
            "start_s": chunks["start_s"].to_numpy()[best[order]],
        }
    )


def query_chunk_of(chunks: pd.DataFrame, track_id: int, start_s: float = 10.0) -> int:
    """Індекс шматка треку, що починається на start_s (це і є «хук»-запит)."""
    m = chunks[(chunks["track_id"] == track_id) & (chunks["start_s"] == start_s)]
    return int(m.index[0])


def pick_queries(tracks: pd.DataFrame, per_genre: int = 1, seed: int = 7) -> list[int]:
    """По per_genre випадкових треків-запитів з кожного жанру (стабільно за seed)."""
    picked = []
    for _, g in tracks.groupby("genre"):
        picked += g.sample(per_genre, random_state=seed).index.tolist()
    return picked


def precision_at_k(emb: np.ndarray, chunks: pd.DataFrame, tracks: pd.DataFrame, k: int = 5) -> pd.Series:
    """Частка top-k з тим самим жанром, що в запиту; запит — шматок із 10-ї секунди кожного треку."""
    q = chunks.index[chunks["start_s"] == 10.0].to_numpy()
    q_tids = chunks["track_id"].to_numpy()[q]
    tids, scores, _ = track_scores(emb, chunks, emb[q])
    scores[q_tids[:, None] == tids[None, :]] = -np.inf
    top = np.argsort(-scores, axis=1)[:, :k]
    genre = tracks["genre"]
    hit = genre.loc[tids].to_numpy()[top] == genre.loc[q_tids].to_numpy()[:, None]
    per_query = pd.Series(hit.mean(axis=1), index=q_tids)
    return per_query.groupby(genre.loc[q_tids].to_numpy()).mean()
