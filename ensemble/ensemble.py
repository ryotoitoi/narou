from numpy import e
import pandas as pd
import os 

os.makedirs("./ensemble/output", exist_ok=True)

df_10 = pd.read_csv("exp_10/output/submit.csv")
df_11 = pd.read_csv("exp_11/output/submit.csv")
df_14 = pd.read_csv("exp_14/output/submit.csv")
df_15 = pd.read_csv("exp_15/output/submit.csv")
df_20 = pd.read_csv("exp_20/output/submit.csv")
df_21 = pd.read_csv("exp_21/output/submit.csv")

sub_df = pd.concat([df_10, df_11, df_15, df_14,df_20,df_21])

sub_df.groupby("ncode").mean().reset_index().to_csv(f"./ensemble/output/submit.csv", index=False)