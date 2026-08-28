# DA3408 Module 1 - Q4 Capstone Reproducibility Drill

This repo has just what's needed for the Q4 reproducibility drill. Nothing else from
the assignment is in here.

## For Partner B

Reproduce the result using only these steps (no other communication about
environment or data allowed, per the assignment):

```bash
git clone <this repo's url>
cd DA3408-Q4-Capstone
git checkout <commit-hash-partner-a-gives-you>
dvc checkout
mamba env create -f environment.yml
conda activate da3408-assignment1
MLFLOW_TRACKING_URI=<tracking-uri-partner-a-gives-you> python src/q4_train_and_register.py
```

Then log a note on the MLflow run (`q4-partner-a-run` or your own run) saying
whether the metric you got matches Partner A's run within a reasonable tolerance,
or explaining the discrepancy if it doesn't.
