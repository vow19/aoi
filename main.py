import os
import random
import time
import json
import numpy as np
import itertools
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  


from model import build_model, measure_model_resources
from dataset import OurDataset
from utils import split_train_val

# 本地/實驗室環境
# ROOT_CSV = "/home/lafer/Lab/data/train.csv"
# ROOT_IMG = "/home/lafer/Lab/data/train_images"
# NUM_WORKERS = 4

# Kaggle 環境
ROOT_CSV = "/kaggle/input/datasets/lafer2003/aoi-data/train.csv"
ROOT_IMG = "/kaggle/input/datasets/lafer2003/aoi-data/train_images"
OUTPUT_DIR = "/kaggle/working/output"
NUM_WORKERS = 4
# ========================================================
# MAX_EPOCH = 100
# BATCH_SIZE = 64
# LR = 0.05

# 切換模型區
# MODEL_NAME = "resnet50.tv_in1k"           # ResNet50（Baseline）
MODEL_NAME = "mobilenetv2_100.ra_in1k"      # MobileNetV2（目前執行）
# MODEL_NAME = "efficientnet_b0.ra_in1k"    # EfficientNet-B0
# ========================================================

# 各模型獨立資料夾，避免切換模型時覆蓋之前的結果
OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_NAME.replace(".", "_"))
NUMBER_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid Search 訓練輪次
SEARCH_EPOCH = 15

# Grid Search 超參數搜尋網格
PARAM_GRID = {
    "lr": [0.001,0.0001],
    "batch_size": [16, 32],
    "weight_decay": [1e-2, 1e-4]
}
# 正式實驗與搜尋分開；門檻須在 pilot 後、正式實驗前固定。
FORMAL_EPOCH = 30
FORMAL_SEEDS = [42]
SEARCH_SEED = 42
TARGET_F1 = 0.98  # pilot 暫定值

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=generator
    )

def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def cpu_state_dict(model):
    # 保存該 epoch 的獨立權重副本，不讓後續 optimizer 更新覆蓋它
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_epoch(loader, optimizer, model, criterion):
    model.train()
    optimizer.zero_grad()
    total_loss, total_samples = 0.0, 0

    for image, label in tqdm(loader, desc="Training", leave=False):
        image, label = image.to(DEVICE), label.to(DEVICE)

        output = model(image)
        loss = criterion(output, label)
        total_loss += loss.item() * image.size(0)
        total_samples += image.size(0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / total_samples


@torch.no_grad()
def val(loader, model, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss, total_samples = 0.0, 0

    for index, (image, label) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        image, label = image.to(DEVICE), label.to(DEVICE)

        output = model(image)  # [Batch, num_classes]
        if criterion is not None:
            total_loss += criterion(output, label).item() * image.size(0)
            total_samples += image.size(0)

        # calculate accuracy
        preds = torch.argmax(output, dim=1)

        # 存入列表（需先轉至 CPU 並轉為 numpy）
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    ### 計算 Accuracy, Macro Precision, Macro Recall, Macro F1
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUMBER_CLASSES)), average="macro", zero_division=0)
    return acc, precision, recall, f1, (total_loss / total_samples if criterion is not None else None)


# 類別名稱與資料集 Label 0–5 對應
CLASS_NAMES = ["Normal", "Void", "Horizontal", "Vertical", "Edge", "Particle"]


def plot_loss(history, prefix):
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [item["train_loss"] for item in history], label="Train Loss")
    ax.plot(epochs, [item["val_loss"] for item in history], label="Validation Loss")
    ax.set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title=f"{MODEL_NAME} - Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(prefix + "_loss_curve.png", dpi=200)
    plt.close(fig)

@torch.no_grad()
def benchmark_inference(model, repeats=200, warmup=30):
    model.eval()
    image = torch.randn(1, 3, 224, 224, device=DEVICE)
    for _ in range(warmup):
        model(image)
    synchronize()
    latencies = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        model(image)
        synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    # 單張 forward latency；排除讀檔、前處理與 CPU→GPU 搬移。
    return {
        "inference_mean_ms": float(np.mean(latencies)),
        "inference_median_ms": float(np.median(latencies)),
        "inference_p95_ms": float(np.percentile(latencies, 95)),
        "batch1_images_per_second": 1000 / float(np.mean(latencies))
    }


@torch.no_grad()
def save_classification_details(loader, model, prefix):
    model.eval()
    labels, predictions = [], []
    for image, label in loader:
        output = model(image.to(DEVICE))
        labels.extend(label.tolist())
        predictions.extend(output.argmax(dim=1).cpu().tolist())
    class_ids = list(range(NUMBER_CLASSES))
    cm = confusion_matrix(labels, predictions, labels=class_ids)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(prefix + "_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(class_ids, CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(class_ids, CLASS_NAMES)
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"{MODEL_NAME} - Best Validation Confusion Matrix")
    threshold = cm.max() / 2
    for i in range(NUMBER_CLASSES):
        for j in range(NUMBER_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(prefix + "_confusion_matrix.png", dpi=200)
    plt.close(fig)
    report = classification_report(labels, predictions, labels=class_ids,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(prefix + "_classification_report.csv")

# 正式實驗固定 FP32，訓練輪次由 FORMAL_EPOCH 設定，不做 early stopping。
def run_formal_experiments(train_set, val_set, params):
    records = []
    metadata = {
        "model": MODEL_NAME, "params": params, "formal_epochs": FORMAL_EPOCH,
        "formal_seeds": FORMAL_SEEDS, "split_seed": 42, "target_f1": TARGET_F1,
        "target_rule": "first validation epoch with macro F1 >= target",
        "device": DEVICE, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "precision": "FP32", "input_shape": [3, 224, 224],
        "num_workers": NUM_WORKERS, "search_grid": PARAM_GRID,
        "search_epochs": SEARCH_EPOCH,
        "evaluation_scope": "validation; used for tuning, not independent test",
        "timing_scope": "train+validation, including data loading; excluding checkpoints",
        "memory_scope": "peak CUDA allocated during training, excluding validation",
        "compute_scope": "batch=1; Conv2d/Linear only; estimated FLOPs=2*MACs",
        "inference_scope": "FP32 batch=1 forward only; excludes loading and CPU/GPU transfer"
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    for seed in FORMAL_SEEDS:
        set_seed(seed)
        train_loader = make_loader(train_set, params["batch_size"], True, seed)
        val_loader = make_loader(val_set, params["batch_size"], False, seed)
        model = build_model(MODEL_NAME, NUMBER_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])
        prefix = os.path.join(OUTPUT_DIR, f"formal_seed_{seed}")
        best_f1, best_state, best_metrics, best_epoch = -1, None, {}, None
        target_epoch, target_seconds = None, None
        cumulative_seconds = 0.0
        peak_training_memory = 0
        history = []
        for epoch in range(1, FORMAL_EPOCH + 1):
            synchronize()
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            train_loss = train_one_epoch(train_loader, optimizer, model, criterion)
            synchronize()
            train_seconds = time.perf_counter() - start
            if DEVICE == "cuda":
                peak_training_memory = max(peak_training_memory, torch.cuda.max_memory_allocated())
            start = time.perf_counter()
            acc, precision, recall, f1, val_loss = val(val_loader, model, criterion)
            synchronize()
            val_seconds = time.perf_counter() - start
            cumulative_seconds += train_seconds + val_seconds
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "accuracy": acc, "precision_macro": precision,
                "recall_macro": recall, "f1_macro": f1,
                "train_seconds": train_seconds, "val_seconds": val_seconds,
                "train_plus_val_seconds": train_seconds + val_seconds,
                "cumulative_train_plus_val_seconds": cumulative_seconds
            })
            if target_epoch is None and f1 >= TARGET_F1:
                target_epoch, target_seconds = epoch, cumulative_seconds
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                best_state = cpu_state_dict(model)
                best_metrics = {"accuracy": acc, "precision_macro": precision,
                                "recall_macro": recall, "f1_macro": f1}
            print(f"Formal seed={seed}, epoch={epoch}/{FORMAL_EPOCH}, F1={f1:.4f}")
        pd.DataFrame(history).to_csv(prefix + "_history.csv", index=False)
        plot_loss(history, prefix)
        torch.save({"model": cpu_state_dict(model), "model_name": MODEL_NAME,
                    "params": params, "seed": seed, "epoch": FORMAL_EPOCH,
                    "metrics": {"accuracy": acc, "precision_macro": precision,
                                "recall_macro": recall, "f1_macro": f1}},
                   prefix + "_final.pt")
        torch.save({"model": best_state, "model_name": MODEL_NAME, "params": params,
                    "seed": seed, "epoch": best_epoch, "metrics": best_metrics},
                   prefix + "_best.pt")
        final_metrics = {"final_accuracy": acc, "final_precision_macro": precision,
                         "final_recall_macro": recall, "final_f1_macro": f1}
        model.load_state_dict(best_state)
        # 運算量量測在訓練結束後，不影響訓練峰值顯存
        resources = measure_model_resources(model)
        save_classification_details(val_loader, model, prefix)
        row = {
            "seed": seed, **best_metrics, **final_metrics, "best_epoch": best_epoch,
            # 總參數量、MACs 與估算 FLOPs
            **resources,
            "target_reached": target_epoch is not None, "target_epoch": target_epoch,
            "time_to_target_seconds": target_seconds,
            "total_train_seconds": sum(item["train_seconds"] for item in history),
            "mean_train_epoch_seconds": np.mean([item["train_seconds"] for item in history]),
            "total_train_plus_val_seconds": cumulative_seconds,
            "peak_training_allocated_MiB": peak_training_memory / (1024 ** 2) if DEVICE == "cuda" else None,
            "train_batch_size": params["batch_size"],
            "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
            **benchmark_inference(model)
        }
        # 未達門檻記為空值，不能以最後一個 epoch 假裝達標。
        records.append(row)
        pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, "formal_results.csv"), index=False)
        del optimizer, model, best_state
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(records)
    # 達標時間的平均只涵蓋達標 runs，同時附 count，避免隱藏未達標情況。
    # 分類表現採最佳驗證 checkpoint；效率及顯存涵蓋完整 FORMAL_EPOCH。
    metric_layout = [
        ("Performance", "Accuracy", "accuracy", "ratio"),
        ("Performance", "Precision (macro)", "precision_macro", "ratio"),
        ("Performance", "Recall (macro)", "recall_macro", "ratio"),
        ("Performance", "Macro F1", "f1_macro", "ratio"),
        ("Efficiency", "Train time", "total_train_seconds", "seconds"),
        ("Efficiency", "Epoch time (mean train)", "mean_train_epoch_seconds", "seconds"),
        ("Efficiency", "Inference time (mean, batch=1)", "inference_mean_ms", "ms/image"),
        ("Resource", "Parameters", "parameters", "count"),
        ("Resource", "MACs (Conv2d/Linear)", "macs_conv_linear", "MACs/image"),
        ("Resource", "FLOPs (estimate, Conv2d/Linear)", "flops_conv_linear_estimate", "FLOPs/image"),
        ("Resource", "Peak VRAM (training allocated)", "peak_training_allocated_MiB", "MiB"),
    ]
    summary = pd.DataFrame([
        {"category": category, "metric": metric, "value": records[0][key], "unit": unit}
        for category, metric, key, unit in metric_layout
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, "formal_summary.csv"), index=False)
    print(f"正式實驗完成：單一種子、{FORMAL_EPOCH} epochs；分類指標為最佳驗證 checkpoint。")
    print(f"達標次數：{int(results['target_reached'].sum())}/{len(results)}")
    print(summary)



def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # create dateset
    # train -> train / validation (9:1) [先讀取train.csv->將dataframe分成9:1->然後傳入dataset]
    train_df, val_df = split_train_val(ROOT_CSV)
    # 保存固定切分，之後三個模型共用同一份資料名單。
    train_df.to_csv(os.path.join(OUTPUT_DIR, "split_train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "split_val.csv"), index=False)

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

        # 每組 trial 重設同一種子，固定初始化與資料亂數來源。
        set_seed(SEARCH_SEED)
        train_loader = make_loader(train_set, batch_size, True, SEARCH_SEED)
        val_loader = make_loader(val_set, batch_size, False, SEARCH_SEED)

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
        # 記錄最佳 epoch 權重及完整學習曲線。
        best_combo_state = None
        best_combo_epoch = None
        combo_history = []

        # 開始單一組合的訓練輪次
        for epoch in range(SEARCH_EPOCH):
            train_loss = train_one_epoch(train_loader, optimizer, model, criterion)
            acc, precision, recall, f1, val_loss = val(val_loader, model, criterion)

            # 搜尋階段曲線供 pilot 檢查。
            combo_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "accuracy": acc,
                                  "precision": precision, "recall": recall, "f1_macro": f1})
            if f1 > best_combo_f1:
                best_combo_f1 = f1
                # 【Codex 修改】權重與最佳 F1 必須來自同一個 epoch。
                best_combo_state = cpu_state_dict(model)
                best_combo_epoch = epoch + 1
                best_combo_metrics = {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_macro": f1
                }

        # 保存每個 trial 曲線。
        pd.DataFrame(combo_history).to_csv(
            os.path.join(OUTPUT_DIR, f"search_trial_{combo_idx + 1}.csv"), index=False
        )
        print(f"-> 組合 {combo_idx + 1} 最佳 Macro F1: {best_combo_f1:.4f} (Acc: {best_combo_metrics.get('accuracy', 0):.4f})")

        # 記錄此組合的結果供報告比對
        search_records.append({
            **params,
            **best_combo_metrics,
            # 最佳 epoch 只作紀錄，不等同收斂。
            "best_epoch": best_combo_epoch
        })

        # 若超越所有組合的歷史最佳，儲存全域最佳權重
        if best_combo_f1 > best_global_f1:
            best_global_f1 = best_combo_f1
            best_params = params
            best_ckpt = {
                "params": params,
                # 存最佳 epoch，原本這裡會存最後 epoch。
                "model": best_combo_state,
                "epoch": best_combo_epoch,
                "model_name": MODEL_NAME,
                "f1_macro": best_combo_f1,
                "metrics": best_combo_metrics
            }
            torch.save(best_ckpt, os.path.join(OUTPUT_DIR, "grid_search_best.pt"))
            print(f">>> 突破全域最佳紀錄！已更新至 {OUTPUT_DIR}/grid_search_best.pt")

        # 釋放前一個 trial，避免兩個模型同時佔 GPU。
        del optimizer, model, best_combo_state
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # 輸出整理好的比較表格
    results_df = pd.DataFrame(search_records)
    results_csv_path = os.path.join(OUTPUT_DIR, "grid_search_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    print("Grid Search 搜尋完成")
    print(f"最佳參數組合: {best_params}")
    print(f"最高 Macro F1: {best_global_f1:.4f}")
    print(f"完整結果已存入: {results_csv_path}")
    print(results_df.sort_values(by="f1_macro", ascending=False).to_string(index=False))
    run_formal_experiments(train_set, val_set, best_params)


if __name__ == "__main__":
    main()
