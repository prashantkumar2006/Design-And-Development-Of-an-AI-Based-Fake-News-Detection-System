"""
Phase 2: Load and inspect the fake news dataset.
Place Fake.csv and True.csv inside the data/ folder before running this.
"""

import pandas as pd

# 1. Load both CSV files
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# 2. Quick look at the data
print("Fake news shape:", fake_df.shape)
print("Real news shape:", true_df.shape)
print("\nColumns:", fake_df.columns.tolist())
print("\nSample fake article:\n", fake_df.iloc[0])

# 3. Add label column
# Convention: 0 = Fake, 1 = Real
fake_df["label"] = 0
true_df["label"] = 1

# 4. Combine into a single dataset
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# 5. Shuffle the rows (important! right now all fakes are on top, all real below)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Basic checks
print("\nCombined dataset shape:", df.shape)
print("\nLabel distribution:\n", df["label"].value_counts())
print("\nAny missing values?\n", df.isnull().sum())

# 7. Save the combined dataset for the next phase (data cleaning)
df.to_csv("data/combined_dataset.csv", index=False)
print("\nSaved combined_dataset.csv to data/ folder.")