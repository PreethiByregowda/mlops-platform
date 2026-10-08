from datasets import load_dataset
import os

# Load the GoEmotions dataset
ds = load_dataset("go_emotions")

print("Available splits:", ds.keys())

os.makedirs("data", exist_ok=True)

# Save train split
ds["train"].to_pandas().to_csv("data/train.csv", index=False)
print("Saved train.csv")

# Save validation split if exists
if "validation" in ds:
    ds["validation"].to_pandas().to_csv("data/valid.csv", index=False)
    print("Saved valid.csv")
else:
    print("Validation split not found.")

# Save test split if exists
if "test" in ds:
    ds["test"].to_pandas().to_csv("data/test.csv", index=False)
    print("Saved test.csv")
else:
    print("Test split not found.")
