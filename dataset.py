import cv2
import pandas as pd
from pathlib import Path

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class OurDataset(Dataset):
    def __init__(self, root_path, df, val=False):
        self.root_path = root_path
        self.df = df 
        self.val = val

        self.images = []
        self.labels = []

        self.load_data()
        # print(self.images)
        # print(self.labels)
        
        # 定義 transforms (大小正規化、數值正規化、轉tensor)
        if not self.val:
            # 訓練集：包含隨機翻轉資料增強
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 驗證集：純粹縮放與正規化，不使用隨機增強
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        

    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #OpenCV預設BGR 

        # image = PIL.Image.open(image).convert('RGB')

        # print(type(image))
        # print(image.shape)
        # print(image)
        image = self.transform(image)
        return image, label
        # C H W

    def __len__(self):
        return len(self.images)

    def load_data(self):

        for _, row in self.df.iterrows():
            image_path = str(Path(self.root_path) / str(row['ID']))
            label =  int(row['Label'])
            self.images.append(image_path)
            self.labels.append(label)


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    data = OurDataset(root_path="data/train_images", df=df, val=False) # 代表dataset class number = 6
    # print(len(data))
    dataloader = DataLoader(data, batch_size=16, shuffle=False)

    for item in dataloader:
        image, label = item
        # print(f"image: {image}")
        # print(f"label: {label}")
        print(f"images shape: {image.shape}")
        print(f"labels shape: {label.shape}")
        break
        # exit()
        # pass
