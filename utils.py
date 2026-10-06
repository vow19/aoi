import pandas as pd


def split_train_val(root_csv):
    df = pd.read_csv(root_csv)

    # random split dataframe into train and validation (9:1)
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    print(f"train 資料: {len(train_df)}, val 資料: {len(val_df)}")
    return train_df, val_df


if __name__ == "__main__":
    train_df, val_df = split_train_val("./data/train.csv")
    print(train_df)
    print(val_df)