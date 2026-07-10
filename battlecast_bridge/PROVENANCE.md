# Vendored Battlecast assets — provenance

The three JavaScript files in `vendor/` are unmodified static assets served
publicly by **https://battlecast.gg** (fetched 2026-07-10):

| File | Role |
|---|---|
| `mc-worker-CQHYBfRR.js` | Self-contained Monte Carlo combat engine (the site's Web Worker) |
| `spells-B65qMaNs.js` | 2024 SRD monster library (317 statblocks) + spell data |
| `heroes-BRXrKxCp.js` | Hero build generator (12 classes × levels 1–20) |

They are vendored so the research grid runs **locally and reproducibly**
(zero load on battlecast.gg, results independent of site updates). All
credit for the combat engine belongs to the Battlecast developers; the
statblocks are 2024 SRD content (CC-BY-4.0 by Wizards of the Coast).

Used for one purpose: generating *fight-to-the-death* win probabilities to
(a) calibrate the survival-physics guard and (b) measure the "DM mercy" gap
between real-play outcomes and simulated deathmatches. See `run_grid.mjs`
(collection) and `analyze.py` (analysis), and the README's Battlecast
section.

**Ruleset caveat**: Battlecast implements the 2024 SRD; the FIREBALL
training data is 2014-era play. Hero builds are Battlecast's optimized
defaults — the simulator measures an *optimal-play* upper bound.
