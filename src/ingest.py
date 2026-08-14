import pandas as pd

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_data()

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== CHURN DISTRIBUTION =====")
    print(df["Churn"].value_counts())

    print("\n===== CHURN PERCENTAGE =====")
    print(df["Churn"].value_counts(normalize=True) * 100)
