# Experiments

The first iteration configures experiments via CLI arguments so an unconsumed configuration file
cannot silently diverge from the actual run. `results/<run>/split.json` captures the relevant
parameters and station-disjoint split for each run. A YAML launcher may be added later.
