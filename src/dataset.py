from torch.utils.data import Dataset
import cv2
from pathlib import Path
import torch

class ImageRestorationDataset(Dataset):
    #提供索引，方便后续getitem检索图片并处理
    def __init__(self,image_dir):
        image_dir = Path(image_dir)
        self.image_paths = list(
            image_dir.glob("*.jpeg")
        )
        #对图片路径进行排序，确保按顺序加载
        self.image_paths.sort()
        

    #返回数据集的长度
    def __len__(self):
        return len(self.image_paths)
    
    #根据索引返回对应的图片,并进行处理，将opencv图片转化为tensor格式
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        clean = cv2.imread(str(image_path))#使用str转换为字符串会更稳定
        if clean is None:
            raise FileNotFoundError("Image not found")
        clean_RGB = cv2.cvtColor(
            clean,
            cv2.COLOR_BGR2RGB
        )
        clean_RGB = cv2.resize(
            clean_RGB,
            (256,256)
        )
        blurred = cv2.GaussianBlur(
            clean_RGB,
            (7,7),
            sigmaX = 2
        )

        clean_image = clean_RGB.transpose(2, 0, 1)
        blurred_image = blurred.transpose(2, 0, 1)

        clean_image = torch.from_numpy(clean_image).float()/255.0
        blurred_image = torch.from_numpy(blurred_image).float()/255.0

        return blurred_image, clean_image

if __name__ == "__main__":
    dataset = ImageRestorationDataset(
        "data/train/clean"
    )
    print("数据集大小:", len(dataset))

    blurred, clean = dataset[0]

    print("Blurred:", blurred.shape)
    print("Clean:", clean.shape)

    print("Blurred dtype:", blurred.dtype)
    print("Clean dtype:", clean.dtype)

    print(
        "Blurred range:",
        blurred.min(),
        blurred.max()
    )

    print(
        "Clean range:",
        clean.min(),
        clean.max()
    )