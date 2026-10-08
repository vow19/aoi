import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_val(root_csv):

    df = pd.read_csv(root_csv)
    # random split dataframe into train and validation (9:1)
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["Label"], random_state=42)

    print(f"train 資料: {len(train_df)}, val 資料: {len(val_df)}")
    return train_df, val_df


if __name__ == "__main__":
    train_df, val_df = split_train_val("./data/train.csv")
    print(train_df)
    print(val_df)
