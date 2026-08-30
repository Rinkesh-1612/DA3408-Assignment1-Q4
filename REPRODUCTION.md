# Reproduction Note

Prabhav Gupta (DA24B018), Partner B

I cloned the repo, checked out ff5e88c and built the environment from
environment.yml then ran dvc checkout to restore the data and then reran the
training script unchanged.

Rinkesh (Partner A) got val_accuracy 0.9639 and val_f1_macro 0.9635.
I got exactly the same. Tolerance 0.001 so it matched.

Run is on the shared MLflow server as q4-partner-b-run, as
q4-digits-classifier v2.
