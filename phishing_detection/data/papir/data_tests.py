import pandas as pd
from _loader_utils import load_mailbench_semicolon_csv, _standardize_cols

# Just peek at standardized columns without building labels/text
df_raw = pd.read_csv("Emails.csv", sep=";", engine="python",
                     quotechar='"', dtype=str, on_bad_lines="warn",
                     encoding="utf-8-sig", header=0, skip_blank_lines=False)
print("Raw columns:", df_raw.columns.tolist())
print("Standardized:", _standardize_cols(df_raw).columns.tolist())
