import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from model import build_model
from dataset import OurDataset
from utils import split_train_val

MAX_EPOCH = 100
BATCH_SIZE = 64
LR = 0.05
MODEL_NAME = "resnet50.tv_in1k"
NUMBER_CLASSES = 6
ROOT_CSV = "./data/train.csv"
NUM_WORKERS = 8  # 8~12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(loader, optimizer, model, criterion):
    model.train()
    optimizer.zero_grad()


    for index, (image, label) in enumerate(tqdm(loader)):
        image, label = image.to(DEVICE), label.to(DEVICE)

        output = model(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def val(loader, model, criterion):
    model.eval()
    correct = 0
    total = 0

    for index, (image, label) in enumerate(tqdm(loader)):
        image, label = image.to(DEVICE), label.to(DEVICE)

        output = model(image)  # [Batch, num_classes]
        # loss = criterion(output, label)

        # calculate accuracy
        preds = torch.argmax(output, dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    acc = correct / total
    print(f"Val Accuracy: {acc:.4f}")
    return acc

def main():
    # create dateset
    # train -> train / validation (9:1) [先讀取train.csv->將dataframe分成9:1->然後傳入dataset]
    train_df, val_df = split_train_val(ROOT_CSV)

    train_set = OurDataset(root_path="./data/train_images", df=train_df, val=False)
    val_set = OurDataset(root_path="./data/train_images", df=val_df, val=True)

    print("dataset建立完畢")
    print(f"train 資料: {len(train_set)}, val 資料: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # create model
    model = build_model(model_name=MODEL_NAME, num_classes=NUMBER_CLASSES)
    model.to(DEVICE)

    # create loss
    criterion = nn.CrossEntropyLoss()

    # learning rate
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    best_acc = -1
    for epoch in range(MAX_EPOCH):
        # train
        train(train_loader, optimizer, model, criterion)

        # validation
        acc = val(val_loader, model, criterion)

        if acc > best_acc:
            best_acc = acc
            # save model
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

        if acc > best_acc:
            best_acc = acc
            # save model
            os.makedirs("./output", exist_ok=True)
            torch.save(ckpt, "./output/best.pt")

        torch.save(ckpt, f"./output/epoch_{epoch}.pt")


if __name__ == "__main__":
    main()