# DA3408 Assignment 1 — Question 4 (Reproducibility Drill)

This is the shared repo Prabhav and I are using for Question 4. It only has the Q4
stuff in it — none of our other assignment work is here.

- **Partner A:** Rinkesh, DA24B017
- **Partner B:** Prabhav, DA24B018

We're each committing our own half, so you can tell who did what from the commit
history.

---

## What's in the repo

| Path | What it does |
|---|---|
| `src/prepare_q4_data.py` | Builds `data/q4_digits.csv` from scikit-learn's bundled digits dataset |
| `src/q4_train_and_register.py` | Trains the model, logs everything to MLflow, registers it and moves it to Staging |
| `data/q4_digits.csv.dvc` | The DVC pointer — the actual CSV isn't in git |
| `dvcstore/` | The DVC cache/remote, kept inside the repo so `dvc checkout` works for anyone who clones |
| `environment.yml` | Pinned versions |

The dataset is 1797 rows plus a header. It comes straight from
`sklearn.datasets.load_digits`, so it's identical on any machine.

---

## Setting it up

```bash
git clone https://github.com/Rinkesh-1612/DA3408-Assignment1-Q4.git
# or over SSH:  git clone git@github.com:Rinkesh-1612/DA3408-Assignment1-Q4.git
cd DA3408-Assignment1-Q4

mamba env create -f environment.yml     # conda works too
conda activate da3408-assignment1

dvc checkout                            # gets data/q4_digits.csv
```

`dvc checkout` works straight after cloning — no credentials and no server needed,
because the DVC cache is committed inside the repo (`dvcstore/`). If it ever gives you
trouble, you can regenerate the data instead and DVC will confirm the hash matches:

```bash
python src/prepare_q4_data.py
dvc status          # should say "Data and pipelines are up to date."
```

### One gotcha with MLflow

The script registers a model and moves it to Staging, and the model registry doesn't
work against a plain `./mlruns` folder — it needs a database behind it. So start the
server with a sqlite backend first:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 127.0.0.1 --port 5000
```

and then point the script at it:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/q4_train_and_register.py
```

I lost a bit of time to this one, so it's worth setting up before you run anything.

---

## My part (Partner A)

I trained a RandomForest on the DVC-versioned dataset and logged the params, the
metrics, the seed and a `git_commit` tag to MLflow, along with the model artifact.
Then I registered it as `q4-digits-classifier` and moved version 1 to **Staging**.

I got `val_accuracy = 0.9639` and `val_f1_macro = 0.9635` with seed 42.

The training script and the `.dvc` file went in as a single commit, which is what the
protocol asks for.

---

## Prabhav's part (Partner B)

Reproduce it using only `git clone`, `git checkout`, `dvc checkout`,
`mamba env create -f environment.yml`, and rerunning the script:

```bash
git clone https://github.com/Rinkesh-1612/DA3408-Assignment1-Q4.git
cd DA3408-Assignment1-Q4
git checkout partner-a-baseline
dvc checkout
mamba env create -f environment.yml
conda activate da3408-assignment1
export MLFLOW_TRACKING_URI=<your MLflow server>
python src/q4_train_and_register.py
```

Then two things:

1. **Leave a note on the MLflow run** saying whether your accuracy came out the same as
   my 0.9639, within whatever tolerance you want to state, or explaining the gap if it
   didn't. You can use the Description box on the run page, or
   `mlflow.set_tag("partner_b_note", "...")`.
2. **Add a `REPRODUCTION.md`** with the number you got, the tolerance, and whether it
   matched. Commit it under your own name so it's clear which half is yours.

---

## For the evaluators

Everything needed to run this is in the repo. `dvc checkout` reads from `dvcstore/` in the
repo itself, and if that fails the dataset regenerates exactly with
`python src/prepare_q4_data.py` — the md5 will still match `data/q4_digits.csv.dvc`.
No external bucket, server or credentials involved.
