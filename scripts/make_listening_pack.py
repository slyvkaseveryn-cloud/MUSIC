"""Збирає listening/index.html: запити-хуки і top-5 для CLAP та ручних ознак.

Запуск:  .venv/bin/python scripts/make_listening_pack.py --subset-size 2000
Кліпи (по 10 с) кладе в listening/audio/, сторінка працює офлайн із Finder.
"""
import argparse
import html
import json

import librosa
import soundfile as sf

from hooklib import CHUNK_S, METHODS, ROOT, load_cache, pick_queries, query_chunk_of, top_k, track_path

OUT = ROOT / "listening"
CLIP_SR = 44100

PAGE = """<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hook Finder — прослуховування (етап 0)</title>
<style>
body{font:15px/1.45 -apple-system,system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#222;background:#fafafa}
h1{font-size:22px} h2{font-size:18px;margin:36px 0 8px;border-top:2px solid #ccc;padding-top:16px}
h3{font-size:15px;margin:18px 0 6px;color:#555}
.q{background:#eef3ff;border-radius:8px;padding:10px 12px}
.r{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px 12px;margin:6px 0}
.meta{color:#666;font-size:13px} audio{width:100%;height:32px;margin:4px 0}
label{margin-right:14px} .sum{font-weight:600} textarea{width:100%;height:120px;font:12px monospace}
button{font:inherit;padding:6px 12px}
</style></head><body>
<h1>Hook Finder — прослуховування результатів (етап 0)</h1>
<p>Для кожного запиту нижче — фрагмент-«хук» (10 с) і по 5 знайдених треків двома методами:
<b>CLAP</b> (нейромережеві ембединги) та <b>ручні ознаки</b> (MFCC, chroma, темп тощо) для порівняння.
У результатах грає той 10-секундний шматок треку, який виявився найближчим до хука.
Послухай і познач, чи звучить схоже. Критерій готовності етапу: для більшості запитів ≥2–3 з 5 справді схожі.</p>
<p class="meta">База: __N__ треків fma_small. Оцінки зберігаються в цьому браузері. Аудіо — з Free Music Archive, ліцензії вказано біля кожного треку.</p>
__BODY__
<h2>Підсумок</h2>
<p id="total" class="sum"></p>
<p><button onclick="exportMarks()">Показати оцінки текстом</button> — скопіюй і надішли Claude.</p>
<textarea id="out" readonly></textarea>
<script>
const KEY='hookfinder-stage0-marks';
let marks={}; try{marks=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(marks))}catch(e){}}
function refresh(){
  const tot={};
  document.querySelectorAll('.block').forEach(b=>{
    let yes=0,done=0;
    b.querySelectorAll('.r').forEach(r=>{const v=marks[r.dataset.id]; if(v){done++; if(v==='так')yes++}});
    b.querySelector('.sum').textContent='схожих: '+yes+' з 5 (оцінено '+done+')';
    const m=b.dataset.method; tot[m]=tot[m]||[0,0]; tot[m][0]+=yes; tot[m][1]+=done;
  });
  document.getElementById('total').textContent=Object.entries(tot).map(([m,v])=>m+': схожих '+v[0]+' з '+v[1]+' оцінених').join(' · ');
}
document.querySelectorAll('.r').forEach(r=>{
  r.querySelectorAll('input').forEach(i=>{
    if(marks[r.dataset.id]===i.value)i.checked=true;
    i.addEventListener('change',()=>{marks[r.dataset.id]=i.value;save();refresh()});
  });
});
function exportMarks(){
  const lines=[];
  document.querySelectorAll('.block').forEach(b=>{
    const v=[...b.querySelectorAll('.r')].map(r=>marks[r.dataset.id]||'?');
    lines.push(b.dataset.query+' | '+b.dataset.method+' | '+v.join(', '));
  });
  document.getElementById('out').value=lines.join('\\n');
}
refresh();
</script></body></html>
"""


def cut_clip(track_id: int, start_s: float) -> str:
    """Вирізає 10 с у listening/audio/ і повертає відносний шлях."""
    name = f"audio/{int(track_id):06d}_{int(start_s):02d}.mp3"
    dst = OUT / name
    if not dst.exists():
        y, _ = librosa.load(track_path(track_id), sr=CLIP_SR, mono=True, offset=start_s, duration=CHUNK_S)
        sf.write(dst, y, CLIP_SR, format="MP3")
    return name


def describe(row) -> str:
    e = lambda v: html.escape(str(v))
    return f"<b>{e(row.artist)}</b> — {e(row.title)}<br><span class='meta'>жанр: {e(row.genre)} · ліцензія: {e(row.license)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-size", type=int, default=2000)
    args = ap.parse_args()

    c = load_cache(args.subset_size)
    tracks, chunks = c["tracks"], c["chunks"]
    (OUT / "audio").mkdir(parents=True, exist_ok=True)

    body, summary = [], []
    for n, qid in enumerate(pick_queries(tracks), 1):
        q = tracks.loc[qid]
        qc = query_chunk_of(chunks, qid)
        start = chunks["start_s"].iloc[qc]
        qname = f"Запит {n} ({q.genre})"
        body.append(
            f"<h2>{html.escape(qname)}</h2><div class='q'>{describe(q)} · фрагмент {int(start)}–{int(start + CHUNK_S)} с · track_id {qid}</span>"
            f"<audio controls preload='none' src='{cut_clip(qid, start)}'></audio></div>"
        )
        for key, label in METHODS.items():
            res = top_k(c[key], chunks, qc)
            body.append(f"<div class='block' data-method='{label}' data-query='{html.escape(qname)}'><h3>{label}: top-5 · <span class='sum'></span></h3>")
            for rank, r in enumerate(res.itertuples(), 1):
                t = tracks.loc[r.track_id]
                rid = f"{qid}-{key}-{r.track_id}"
                body.append(
                    f"<div class='r' data-id='{rid}'>{rank}. {describe(t)} · подібність: {r.score:.3f} · шматок {int(r.start_s)}–{int(r.start_s + CHUNK_S)} с</span>"
                    f"<audio controls preload='none' src='{cut_clip(r.track_id, r.start_s)}'></audio>"
                    f"схоже? <label><input type='radio' name='{rid}' value='так'> так</label>"
                    f"<label><input type='radio' name='{rid}' value='ні'> ні</label></div>"
                )
                summary.append({"query": int(qid), "method": key, "rank": rank, "track_id": int(r.track_id), "score": round(float(r.score), 4)})
            body.append("</div>")

    (OUT / "index.html").write_text(PAGE.replace("__N__", str(len(tracks))).replace("__BODY__", "\n".join(body)), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"готово: {OUT / 'index.html'}, кліпів: {len(list((OUT / 'audio').glob('*.mp3')))}")


if __name__ == "__main__":
    main()
