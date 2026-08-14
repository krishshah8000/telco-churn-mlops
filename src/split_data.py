import os
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

INITIAL_DATA_PATH = "data/initial/initial_data.csv"
NEW_DATA_PATH = "data/new/new_data.csv"
HELD_OUT_DATA_PATH = "data/held_out/held_out_data.csv"


def split_dataset():
    df = pd.read_csv(RAW_DATA_PATH)

    # First split:
    # 80% temporary data, 20% held-out data
    temp_data, held_out_data = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["Churn"]
    )

    # Second split:
    # From the remaining 80%:
    # 75% → initial = 60% of total
    # 25% → new = 20% of total
    initial_data, new_data = train_test_split(
        temp_data,
        test_size=0.25,
        random_state=42,
        stratify=temp_data["Churn"]
    )

    os.makedirs("data/initial", exist_ok=True)
    os.makedirs("data/new", exist_ok=True)
    os.makedirs("data/held_out", exist_ok=True)

    initial_data.to_csv(INITIAL_DATA_PATH, index=False)
    new_data.to_csv(NEW_DATA_PATH, index=False)
    held_out_data.to_csv(HELD_OUT_DATA_PATH, index=False)

    print("\n===== DATA SPLIT COMPLETE =====")
    print(f"Initial data : {len(initial_data)} rows")
    print(f"New data     : {len(new_data)} rows")
    print(f"Held-out data: {len(held_out_data)} rows")
    print(f"Total        : {len(initial_data) + len(new_data) + len(held_out_data)} rows")

    print("\n===== CHURN DISTRIBUTION =====")

    print("\nInitial data:")
    print(initial_data["Churn"].value_counts())

    print("\nNew data:")
    print(new_data["Churn"].value_counts())

    print("\nHeld-out data:")
    print(held_out_data["Churn"].value_counts())


if __name__ == "__main__":
    split_dataset()
