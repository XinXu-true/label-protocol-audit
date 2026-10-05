# Label-protocol audit for EEG-based emotion recognition

Code accompanying the manuscript *"Label protocols as measurement instruments:
how much do reported emotion-recognition numbers depend on the conversion of
self-reports to labels"* (under review).

Every study of EEG-based emotion recognition turns each participant's
self-report rating into a training label with some rule — a fixed threshold, a
threshold that discards ambiguous middle ratings, or a within-subject
standardisation. That rule, which we call the **label protocol**, decides which
trials enter training with which class, and it is almost never reported.

This repository measures what the choice costs. It reads only the rating
columns of the [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset,
trains no model, and answers three questions:

1. **How much do the protocols disagree?** Flip rate, discarded share and class
   balance shift, computed on the same trials for all three protocols across the
   four DEAP rating dimensions.
2. **How far apart can reported accuracies be?** For any model, the accuracies
   computed under two protocols can differ by at most the label flip rate
   between them. Measuring the flip rate therefore bounds the incomparability of
   two reported numbers without training anything.
3. **Where does the disagreement come from?** Flips concentrate near the
   threshold and grow with a participant's anchor offset from the scale midpoint.

## Requirements

- Python 3.11 or later
- `numpy`, `scipy`, `matplotlib` (see `requirements.txt`); `pytest` for the tests

```bash
pip install -r requirements.txt
```

## Quick start: no dataset required

```bash
python scripts/demo_simulated.py
```

This runs the whole audit on synthetic 1–9 ratings (32 participants × 40 trials,
the DEAP shape) and checks that the structural facts hold: the discard variant
leaves every retained label unchanged, it removes roughly a fifth of the trials,
and flips concentrate near the threshold.

## Running the audit on DEAP

DEAP is distributed by Queen Mary University of London and requires a request
form: <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>. Point the scripts at the
`data_preprocessed_python` directory of your copy, either by placing it at the
default location or by setting an environment variable:

```bash
export DEAP_ROOT=/path/to/data_preprocessed_python     # or set DEAP_ROOT on Windows

python scripts/run_label_audit.py   # writes results/label_audit.json
python scripts/run_mechanism.py     # writes results/mechanism.json
python scripts/make_figures.py      # writes results/figs/fig1_protocols.*, fig2_mechanism.*
pytest tests/                       # unit tests for the protocol core
```

Each script takes a few seconds. Nothing is sampled and nothing is trained, so
repeated runs produce identical output. The one randomised step is the
participant-level bootstrap that accompanies each corpus-level rate; it uses a
fixed seed (0) and 10,000 resamples, and is reproducible for that reason.

## The protocols

| Name | Rule | Note |
|---|---|---|
| **P0** | rating above 5 is high, otherwise low | a rating of exactly 5 is low, matching the LibEER benchmark's `<=` convention |
| **P1** | ratings ≤ 4 low, ≥ 6 high, the 4–6 band discarded | the discarded trials leave the evaluation set rather than changing labels |
| **P2** | standardise each participant's ratings by their own mean and SD, take the sign | falls back to the group SD when a participant's SD is near zero |

A fourth variant is computed as a worked example of cross-paper variation: the
threshold of three used for valence binarisation in a published transformer
study, with ratings above three in the high class.

P2 does **not** give every participant a class share of exactly one half. On an
integer scale the count of ratings at or above a participant's own mean is not
in general half the trials; the shares span 30.0 to 87.5 percent across the four
dimensions. What P2 does is weaken the dependence of class balance on where a
participant's scale is anchored (Pearson *r* falls from 0.84–0.93 to 0.13–0.62)
and roughly halve the between-participant spread of that balance.

## Repository layout

```
src/prlpaper/labels/audit.py    the three protocols and the disagreement measures
src/prlpaper/data/deap.py       DEAP loader (ratings and windows)
src/prlpaper/protocol/split.py  window/split utilities used by the loader
scripts/                        audit, mechanism, figure and self-check entry points
tests/test_audit.py             tie rule, pairing semantics, P2 behaviour on ties
results/                        generated outputs (JSON) and the two figures
```

The repository contains only the zero-training audit. Exploratory code from
earlier stages of the project is not included.

## Citation

If you use this code, please cite the manuscript:

> Xu, X. (2026). *Label protocols as measurement instruments: how much do
> reported emotion-recognition numbers depend on the conversion of self-reports
> to labels.* Manuscript under review.

The dataset should be cited as:

> Koelstra, S., Mühl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T.,
> Pun, T., Nijholt, A., & Patras, I. (2012). DEAP: A database for emotion
> analysis using physiological signals. *IEEE Transactions on Affective
> Computing, 3*(1), 18–31. <https://doi.org/10.1109/T-AFFC.2011.15>

## License

MIT — see `LICENSE`.
