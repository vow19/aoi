import os
import itertools
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  ### 【新增指標】：引入評估工具

from model import build_model
from dataset import OurDataset
from utils import split_train_val

# ==================== 【本地/實驗室環境】 ====================
# ROOT_CSV = "./data/train.csv"
# ROOT_IMG = "./data/train_images"
# OUTPUT_DIR = "./output"
# NUM_WORKERS = 8

# 【Kaggle 環境】
ROOT_CSV = "/kaggle/input/datasets/lafer2003/aoi-data/train.csv"
ROOT_IMG = "/kaggle/input/datasets/lafer2003/aoi-data/train_images"
OUTPUT_DIR = "/kaggle/working/output"
NUM_WORKERS = 4
# ========================================================

# MAX_EPOCH = 100
# BATCH_SIZE = 64
# LR = 0.05

MODEL_NAME = "resnet50.tv_in1k"
NUMBER_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid Search 訓練輪次（網格搜尋通常為了控制時間，設為 10~20 epoch）
SEARCH_EPOCH = 15

# Grid Search 超參數搜尋網格
PARAM_GRID = {
    "lr": [0.0001​],
    "batch_size": [16,32],
    "weight_decay": [1e-2, 1e-4]
}

def train_one_epoch(loader, optimizer, model, criterion):
    model.train()
    optimizer.zero_grad()

    for image, label in tqdm(loader, desc="Training", leave=False):
        image, label = image.to(DEVICE), label.to(DEVICE)

        output = model(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def val(loader, model, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []

    for index, (image, label) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        image, label = image.to(DEVICE), label.to(DEVICE)

        output = model(image)  # [Batch, num_classes]
        # loss = criterion(output, label)

        # calculate accuracy
        preds = torch.argmax(output, dim=1)

        # 存入列表（需先轉至 CPU 並轉為 numpy）
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    ### 計算 Accuracy, Macro Precision, Macro Recall, Macro F1
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return acc, precision, recall, f1


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # create dateset
    # train -> train / validation (9:1) [先讀取train.csv->將dataframe分成9:1->然後傳入dataset]
    train_df, val_df = split_train_val(ROOT_CSV)

    train_set = OurDataset(root_path=ROOT_IMG, df=train_df, val=False)
    val_set = OurDataset(root_path=ROOT_IMG, df=val_df, val=True)

    # print("dataset建立完畢")
    # print(f"train 資料: {len(train_set)}, val 資料: {len(val_set)}")

    # 將參數字典條列窮舉所有組合 (Grid Search)
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*PARAM_GRID.values()))
    print(f"開始 Grid Search，共計 {len(combinations)} 組參數")

    best_global_f1 = -1
    best_params = None
    search_records = []

    for combo_idx, values in enumerate(combinations):
        params = dict(zip(keys, values))
        lr = params["lr"]
        batch_size = params["batch_size"]
        weight_decay = params["weight_decay"]

        print(f"\n[{combo_idx + 1}/{len(combinations)}] 正在測試參數組合: {params}")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        # create model
        model = build_model(model_name=MODEL_NAME, num_classes=NUMBER_CLASSES)
        model.to(DEVICE)

        # create loss
        criterion = nn.CrossEntropyLoss()

        # learning rate
        # optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_combo_f1 = -1
        best_combo_metrics = {}

        # 開始單一組合的訓練輪次
        for epoch in range(SEARCH_EPOCH):
            train_one_epoch(train_loader, optimizer, model, criterion)
            acc, precision, recall, f1 = val(val_loader, model, criterion)

            if f1 > best_combo_f1:
                best_combo_f1 = f1
                best_combo_metrics = {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_macro": f1
                }

        print(f"-> 組合 {combo_idx + 1} 最佳 Macro F1: {best_combo_f1:.4f} (Acc: {best_combo_metrics.get('accuracy', 0):.4f})")

        # 記錄此組合的結果供報告比對
        search_records.append({
            **params,
            **best_combo_metrics
        })

        # 若超越所有組合的歷史最佳，儲存全域最佳權重
        if best_combo_f1 > best_global_f1:
            best_global_f1 = best_combo_f1
            best_params = params
            best_ckpt = {
                "params": params,
                "model": model.state_dict(),
                "f1_macro": best_combo_f1,
                "metrics": best_combo_metrics
            }
            torch.save(best_ckpt, os.path.join(OUTPUT_DIR, "grid_search_best.pt"))
            print(f">>> 突破全域最佳紀錄！已更新至 {OUTPUT_DIR}/grid_search_best.pt")

    # 輸出整理好的比較表格
    results_df = pd.DataFrame(search_records)
    results_csv_path = os.path.join(OUTPUT_DIR, "grid_search_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    print("\n" + "=" * 50)
    print("Grid Search 搜尋完成")
    print(f"最佳參數組合: {best_params}")
    print(f"最高 Macro F1: {best_global_f1:.4f}")
    print(f"完整結果已存入: {results_csv_path}")
    print("=" * 50)
    print(results_df.sort_values(by="f1_macro", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()