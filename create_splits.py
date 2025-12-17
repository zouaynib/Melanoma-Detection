import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    PROC_ROOT = Path("processed")
    META_PATH = PROC_ROOT / "all_metadata.csv"

    df = pd.read_csv(META_PATH)
    print(f"Loaded metadata: {df.shape[0]} rows")

    # ======================================================
    # Convert diagnosis string → integer label
    # ======================================================
    class_names = ["AK", "BCC", "BKL", "DF", "NV", "MEL", "SCC", "VASC"]
    label2id = {c: i for i, c in enumerate(class_names)}
    df["label"] = df["diagnosis"].map(label2id)

    # ======================================================
    # Optional: clean metadata if needed
    # Fill missing age with median
    # Fill missing sex & site with "unknown"
    # ======================================================
    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())
    df["sex"] = df["sex"].fillna("unknown")
    df["anatom_site_general"] = df["anatom_site_general"].fillna("unknown")

    # ======================================================
    # Split: 70% train, 15% val, 15% test
    # Stratify using class labels
    # ======================================================
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ======================================================
    # Save splits
    # ======================================================
    SPLIT_DIR = PROC_ROOT / "splits"
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print("Saved train/val/test splits → processed/splits/")


if __name__ == "__main__":
    main()
