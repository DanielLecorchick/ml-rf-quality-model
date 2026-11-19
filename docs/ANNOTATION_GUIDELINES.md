# Initial manual quality-labeling guidelines (DRAFT)

* `0` — clearly usable observation for its intended scientific analysis.
* `1` — clearly unusable due to missing samples, destructive RF interference, receiver malfunction,
  severe clipping, or known acquisition failure, based on expert review.
* blank — ambiguous quality; **do not force a binary label**.

Do not use absence of a strong spectral line or Doppler variation alone as evidence of a faulty
station. Consider known reference transmitter schedules, site-specific propagation and time of day.

Record reviewer, station, date, reason, source spectrogram, and observational context outside the
simple demonstration CSV. Obtain independently reviewed examples for the final benchmark.

### Visual annotation workflow

Run `rf-quality review-gallery` and open its generated `index.html` in a browser.
The export button downloads a CSV compatible with `rf-quality apply-labels`.
Do not label by reading a single spectrogram alone: retain contextual metadata,
instrument calibration, expected transmitters, time of day and reviewer rationale.
The gallery samples stations evenly; it is not a substitute for stratified
labeled sample planning or double-review of an independent test dataset.
