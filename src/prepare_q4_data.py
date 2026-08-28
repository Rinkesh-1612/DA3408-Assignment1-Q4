"""Q4 capstone: materialize a small, fixed dataset for the reproducibility drill."""
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()
df = pd.DataFrame(digits.data, columns=[f"px{i}" for i in range(digits.data.shape[1])])
df["target"] = digits.target
df.to_csv("data/q4_digits.csv", index=False)
print(f"Wrote data/q4_digits.csv with {len(df)} rows")
