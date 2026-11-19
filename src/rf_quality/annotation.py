"""Offline, browser-based annotation gallery with spectrograms and CSV export.

No web server or network access required. The user must explicitly export
review.csv before exiting; annotations are not uploaded anywhere.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .dataset import WindowDataset


def select_review_indices(ds: WindowDataset, max_windows: int, seed: int = 42) -> list[int]:
    """Sample across stations (round robin) rather than only first downloaded station."""
    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    rng = np.random.default_rng(seed)
    stations = sorted(set(map(str, ds.station_ids)))
    groups = [list(map(int, rng.permutation(np.flatnonzero(ds.station_ids == station))))
              for station in stations]
    chosen = []
    while len(chosen) < min(max_windows, len(ds.labels)):
        active = False
        for group in groups:
            if group and len(chosen) < max_windows:
                chosen.append(group.pop())
                active = True
        if not active:
            break
    return chosen


def _plot_observation(iq: np.ndarray, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram, welch

    z = iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)
    power = np.abs(z) ** 2
    _, psd = welch(z, fs=1.0, nperseg=min(256, len(z)), return_onesided=False)
    freq, times, spectral = spectrogram(z, fs=1.0, nperseg=min(128, len(z)),
                                         noverlap=min(128, len(z)) // 2,
                                         return_onesided=False, mode="psd")
    order = np.argsort(freq)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), constrained_layout=True)
    axes[0].plot(iq[0], label="I", linewidth=.65)
    axes[0].plot(iq[1], label="Q", linewidth=.65, alpha=.75)
    axes[0].plot(np.sqrt(power), linewidth=.6, label="|IQ|", alpha=.75)
    axes[0].set(xlabel="Sample", ylabel="I/Q amplitude")
    axes[0].legend(loc="upper right", ncol=3, fontsize=7)
    axes[1].plot(np.fft.fftshift(np.fft.fftfreq(len(psd))),
                 10 * np.log10(np.maximum(np.fft.fftshift(psd.real), 1e-14)), lw=.8)
    axes[1].set(xlabel="Frequency (cycles/sample)", ylabel="PSD (dB, relative)")
    axes[2].pcolormesh(times, freq[order],
                       10 * np.log10(np.maximum(spectral[order].real, 1e-14)),
                       shading="auto", cmap="viridis")
    axes[2].set(xlabel="Time (sample indices / sample rate)",
                ylabel="Frequency (cycles/sample)")
    fig.savefig(output, dpi=110)
    plt.close(fig)


def build_review_gallery(ds: WindowDataset, output: Path, *, max_windows: int = 100,
                         seed: int = 42) -> Path:
    """Write `index.html`, plots and a compatible CSV template; return HTML path."""
    ds.validate()
    output = Path(output)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    from .labels import create_label_template
    create_label_template(ds, output / "review_template_all.csv")
    indices = select_review_indices(ds, max_windows, seed)
    entries = []
    for idx in indices:
        identifier = str(ds.window_ids[idx])
        key = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        name = f"{key}.png"
        _plot_observation(ds.signals[idx], plots / name)
        entries.append({
            "window_id": identifier,
            "station_id": str(ds.station_ids[idx]),
            "observation_id": str(ds.observation_ids[idx]),
            "timestamp_utc_seconds": (None if ds.timestamps is None else float(ds.timestamps[idx])),
            "initial_label": (None if ds.labels[idx] == -1 else int(ds.labels[idx])),
            "plot": "plots/" + name,
        })
    (output / "review_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    dataset_id = hashlib.sha256("|".join(map(str, ds.window_ids)).encode()).hexdigest()[:16]
    cards = []
    for n, entry in enumerate(entries):
        title = html.escape(entry["station_id"] + " — " + entry["window_id"])
        img = html.escape(entry["plot"], quote=True)
        cards.append(f'''<article class="card" data-index="{n}">
  <h2>{n + 1}. {title}</h2>
  <img alt="I/Q waveform, power spectrum and time-frequency plot" src="{img}">
  <div class="choices">
    <label><input type="radio" name="label-{n}" value="0"> Acceptable (0)</label>
    <label><input type="radio" name="label-{n}" value="1"> Poor (1)</label>
    <label><input type="radio" name="label-{n}" value=""> Uncertain / skip</label>
  </div>
  <label>Reason <select id="reason-{n}"><option value="">—</option>
    <option>interference</option><option>dropout</option><option>clipping</option>
    <option>receiver issue</option><option>clean signal</option><option>other</option>
  </select></label>
  <label>Notes <input id="notes-{n}" placeholder="What makes this observation usable or unusable?"></label>
</article>''')
    # Embed as JSON with HTML metacharacters escaped to prevent tag injection.
    records_json = json.dumps(entries).replace("<", "\\u003c").replace(">", "\\u003e")
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>PSWS RF Quality — local annotation</title>
<style>body{font:15px system-ui;margin:2rem auto;max-width:1000px;padding:0 1rem;background:#f7f8fa;color:#162333}
.card{background:white;border:1px solid #d4dce4;border-radius:9px;margin:1.5rem 0;padding:1rem}
.card img{max-width:100%;width:100%;object-fit:contain}h1{font-size:1.6rem}h2{font-size:1rem;overflow-wrap:anywhere}
.choices{display:flex;gap:2rem;flex-wrap:wrap;margin:1rem 0}button{padding:.6rem 1rem;cursor:pointer}
input[type=text],input:not([type]),select{padding:.4rem} .card label{margin-right:1rem}
header{background:white;border:1px solid #d4dce4;padding:1.2rem;border-radius:9px}</style>
</head><body><header><h1>PSWS RF quality review — local only</h1>
<p><strong>0</strong>: usable; <strong>1</strong>: unusable; <strong>uncertain</strong>: do not force a label.
Plots show normalized frequency (cycles/sample), not a calibrated receiver frequency.
A legitimate ionospheric disturbance is not automatically bad data.</p>
<p>Only selected examples are shown here. Review all stations and both classes before training.
Annotations remain in this browser until you click export; export replaces prior same-named downloads.</p>
<label>Reviewer <input id="reviewer" placeholder="reviewer ID"></label>
<button id="export">Export review.csv</button> <span id="count"></span>
</header>\n''' + "\n".join(cards) + '''
<script>
const entries = RECORDS;
const storageKey = 'psws_review_DATASET_ID';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) {}
const radio = (n) => document.querySelector(`input[name="label-${n}"]:checked`);
function collect() {
 entries.forEach((entry, n) => {
   saved[entry.window_id] = {label: radio(n)?.value ?? '',
    reason: document.getElementById(`reason-${n}`).value,
    notes: document.getElementById(`notes-${n}`).value};
 });
 saved.__reviewer = document.getElementById('reviewer').value;
 try { localStorage.setItem(storageKey, JSON.stringify(saved)); } catch (_) {}
 update();
}
function update() {
 const labeled = entries.filter(e => ['0','1'].includes(saved[e.window_id]?.label)).length;
 document.getElementById('count').textContent = `${labeled}/${entries.length} labeled`;
}
entries.forEach((e,n) => {
 const item = saved[e.window_id];
 if (item) {
   const v = item.label === '' ? '' : item.label;
   const el = document.querySelector(`input[name="label-${n}"][value="${v}"]`);
   if (el) el.checked = true;
   document.getElementById(`reason-${n}`).value = item.reason || '';
   document.getElementById(`notes-${n}`).value = item.notes || '';
 }
 document.querySelectorAll(`.card[data-index="${n}"] input, .card[data-index="${n}"] select`)
 .forEach(el => el.addEventListener('change', collect));
 document.getElementById(`notes-${n}`).addEventListener('input', collect);
});
document.getElementById('reviewer').value = saved.__reviewer || '';
document.getElementById('reviewer').addEventListener('input', collect);
function csv(x){const v=String(x??'');return '"'+v.replaceAll('"','""')+'"';}
document.getElementById('export').addEventListener('click', () => {
 collect();
 const rows = [['window_id','station_id','observation_id','label','reviewer_notes','reason','reviewer']];
 entries.forEach(e => {const v=saved[e.window_id] || {};
  rows.push([e.window_id,e.station_id,e.observation_id,v.label || '',v.notes || '',v.reason || '',saved.__reviewer || '']);});
 const file = new Blob([rows.map(r=>r.map(csv).join(',')).join('\r\n')+'\r\n'],{type:'text/csv'});
 const url = URL.createObjectURL(file);const link=document.createElement('a');link.href=url;
 link.download='review.csv';link.click();URL.revokeObjectURL(url);
});
update();
</script></body></html>'''
    page = page.replace("RECORDS", records_json).replace("DATASET_ID", dataset_id)
    target = output / "index.html"
    target.write_text(page)
    return target
