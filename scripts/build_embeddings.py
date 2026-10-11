"""Рахує й кешує вектори шматків: CLAP і ручні ознаки (baseline).

Запуск:  .venv/bin/python scripts/build_embeddings.py --subset-size 2000
Повний fma_small:                                      --subset-size 8000
Результат у data/embeddings/: clap_n*.npy, baseline_n*.npy, chunks_n*.csv, tracks_n*.csv
"""
import argparse
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from hooklib import CHUNK_S, EMB_DIR, HOP_S, balanced_subset, l2norm, load_small_metadata, track_path

# laion/larger_clap_music у transformers 5.19 вантажиться без помилок, але працює як зламана
# (zero-shot за жанром на рівні випадкового, logit_scale ~0, precision@5 0.18 проти 0.50 тут),
# тому беремо сусідній музичний чекпойнт. Перевірено на 400 треках fma_small.
CLAP_NAME = "laion/larger_clap_music_and_speech"
CLAP_SR = 48000
FEAT_SR = 22050
MIN_DURATION_S = 25.0


def handcrafted(y: np.ndarray, sr: int) -> np.ndarray:
    """MFCC mean/std, chroma mean, spectral centroid mean/std, contrast mean, темп."""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tempo = librosa.feature.tempo(y=y, sr=sr)
    return np.concatenate(
        [
            mfcc.mean(1), mfcc.std(1), chroma.mean(1),
            [centroid.mean(), centroid.std()], contrast.mean(1), np.atleast_1d(tempo)[:1],
        ]
    ).astype(np.float32)


def load_track(track_id: int):
    """Працює у воркері: декодує mp3, ріже на шматки, рахує ручні ознаки."""
    import librosa

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(track_path(track_id), sr=CLAP_SR, mono=True)
            if len(y) < MIN_DURATION_S * CLAP_SR:
                return track_id, None, f"закороткий: {len(y) / CLAP_SR:.1f} с"
            n, hop = int(CHUNK_S * CLAP_SR), int(HOP_S * CLAP_SR)
            # кліпи FMA часто на долі секунди коротші за 30 с — доповнюємо тишею, щоб не втратити останній шматок
            y = np.pad(y, (0, (-len(y)) % hop if (-len(y)) % hop <= CLAP_SR // 2 else 0))
            starts = np.arange(0, len(y) - n + 1, hop)
            chunks = np.stack([y[s : s + n] for s in starts])
            y22 = librosa.resample(y, orig_sr=CLAP_SR, target_sr=FEAT_SR)
            n22 = int(CHUNK_S * FEAT_SR)
            feats = np.stack(
                [handcrafted(y22[int(s / CLAP_SR * FEAT_SR) :][:n22], FEAT_SR) for s in starts]
            )
        return track_id, (starts / CLAP_SR, chunks, feats), None
    except Exception as e:  # биті mp3 у FMA трапляються — пропускаємо й логуємо
        return track_id, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--model", default=CLAP_NAME)
    args = ap.parse_args()

    import torch
    from transformers import ClapModel, ClapProcessor

    tracks = balanced_subset(load_small_metadata(), args.subset_size, args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"треків: {len(tracks)}, пристрій: {device}, модель: {args.model}")

    t0 = time.time()
    model = ClapModel.from_pretrained(args.model).eval().to(device)
    fe = ClapProcessor.from_pretrained(args.model).feature_extractor
    print(f"модель завантажено за {time.time() - t0:.1f} с")

    def embed(batch: list[np.ndarray]) -> np.ndarray:
        inp = fe(batch, sampling_rate=CLAP_SR, return_tensors="pt")
        with torch.no_grad():
            out = model.get_audio_features(**{k: v.to(device) for k, v in inp.items()})
        return l2norm(out.pooler_output.float().cpu().numpy())

    rows, clap, feats, skipped = [], [], [], []
    t_clap, t0 = 0.0, time.time()
    with Pool(args.workers) as pool:
        for i, (tid, res, err) in enumerate(pool.imap(load_track, tracks.index, chunksize=4), 1):
            if err:
                skipped.append((tid, err))
                continue
            starts, chunks, f = res
            t1 = time.time()
            clap.append(np.concatenate([embed(list(chunks[j : j + args.batch_size])) for j in range(0, len(chunks), args.batch_size)]))
            t_clap += time.time() - t1
            feats.append(f)
            rows += [(tid, float(s)) for s in starts]
            if i % 200 == 0:
                print(f"  {i}/{len(tracks)}  {time.time() - t0:.0f} с", flush=True)
    total = time.time() - t0

    clap = np.concatenate(clap).astype(np.float32)
    raw = np.concatenate(feats)
    # стандартизація кожної ознаки, потім L2 — щоб і тут працювала косинусна подібність
    base = l2norm((raw - raw.mean(0)) / (raw.std(0) + 1e-9)).astype(np.float32)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n{args.subset_size}"
    ok = pd.unique(pd.Series([r[0] for r in rows]))
    np.save(EMB_DIR / f"clap_{tag}.npy", clap)
    np.save(EMB_DIR / f"baseline_{tag}.npy", base)
    np.save(EMB_DIR / f"baseline_raw_{tag}.npy", raw)
    pd.DataFrame(rows, columns=["track_id", "start_s"]).to_csv(EMB_DIR / f"chunks_{tag}.csv", index=False)
    tracks.loc[ok].to_csv(EMB_DIR / f"tracks_{tag}.csv")
    pd.DataFrame(skipped, columns=["track_id", "reason"]).to_csv(EMB_DIR / f"skipped_{tag}.csv", index=False)

    print(f"готово: {len(ok)} треків, {len(rows)} шматків; CLAP {clap.shape}, baseline {base.shape}")
    print(f"час: усього {total:.0f} с ({total / len(tracks):.2f} с/трек), з них CLAP-інференс {t_clap:.0f} с")
    print(f"пропущено: {len(skipped)} {skipped[:10]}")


if __name__ == "__main__":
    main()
