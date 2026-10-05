"""Label-protocol audit for EEG-based emotion recognition.

Companion code for the manuscript "Label protocols as measurement instruments:
how much do reported emotion-recognition numbers depend on the conversion of
self-reports to labels".

The audit reads only the self-report rating columns of the DEAP dataset. No
model is trained anywhere, so every reported quantity is deterministic and can
be recomputed exactly from the released data.

Entry points (run from the repository root):

    python scripts/demo_simulated.py     # self-check on synthetic ratings, no data needed
    python scripts/run_label_audit.py    # three protocols x four rating dimensions
    python scripts/run_mechanism.py      # mechanism bins and failure conditions
    python scripts/make_figures.py       # the two manuscript figures
"""

__version__ = "0.1.0"
