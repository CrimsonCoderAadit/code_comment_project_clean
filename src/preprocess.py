import pandas as pd

def load_and_clean(path="data/Code_Comment_Seed_Data.csv", save_path="data/cleaned_dataset.csv"):
    # Load dataset
    df = pd.read_csv(path)

    # Rename columns for convenience
    df = df.rename(columns={
        "Comments": "comment",
        "Surrounding Code Context": "code",
        "Class": "label"
    })

    # Normalize labels: Not Useful → 0, Useful → 1
    df["label"] = df["label"].map({"Not Useful": 0, "Useful": 1})

    # Drop any rows with missing values
    df = df.dropna()

    # Save cleaned dataset
    df.to_csv(save_path, index=False)

    # Print dataset info
    print("✅ Cleaned dataset saved to:", save_path)
    print("Shape:", df.shape)
    print("\nClass balance:")
    print(df["label"].value_counts())

    print("\nLegend: 0 = Not Useful, 1 = Useful")

    return df

if __name__ == "__main__":
    df = load_and_clean()
    print("\nPreview:")
    print(df.head())
