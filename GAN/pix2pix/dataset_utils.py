from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import config


class MapDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img = self.list_files[index]
        img_path = os.path.join(self.root_dir, img)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600] # images are 1200xsomething
        target_image = image[:, 600:]
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

if __name__ == "__main__":
    dataset = MapDataSet("datasets/maps/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        import sys
        sys.exit()
