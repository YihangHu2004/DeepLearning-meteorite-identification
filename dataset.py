import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def _require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return path


def _load_csv(path, required_columns):
    df = pd.read_csv(_require_file(path))
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    return df


def resolve_train_paths(root):
    image_dir = os.path.join(root, "train_images")
    csv_path = os.path.join(root, "train_labels.csv")
    _require_file(csv_path)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"{image_dir} not found.")
    return image_dir, csv_path


def resolve_test_paths(root):
    image_dir = os.path.join(root, "test_images")
    template_csv_path = os.path.join(root, "sample_submission.csv")
    _require_file(template_csv_path)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"{image_dir} not found.")
    return image_dir, template_csv_path


class StoneDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        """
        root: 数据集根目录，例如 ./dataset
        split: 'train' 或 'test'
        transforms: 图像预处理变换

        当前目录结构应为：
        root/
        |- train_images/
        |- train_labels.csv
        |- test_images/
        |- sample_submission.csv
        """
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

        self.root = root
        self.split = split
        self.transforms = transforms
        self.samples = []
        self.labels = []

        if split == "train":
            self._load_train_samples()
        else:
            self._load_test_samples()

    def _load_train_samples(self):
        image_dir, csv_path = resolve_train_paths(self.root)
        df = _load_csv(csv_path, required_columns=["id", "label"])

        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row["id"])
            self.samples.append(img_path)
            self.labels.append(int(row["label"]))

    def _load_test_samples(self):
        image_dir, template_csv_path = resolve_test_paths(self.root)
        df = _load_csv(template_csv_path, required_columns=["id"])

        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row["id"])
            self.samples.append(img_path)
            self.labels.append(None)

    def __getitem__(self, index):
        img_path = self.samples[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        if self.split == "test":
            return image, img_path
        return image, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, "dataset")
    dataset_train = StoneDataset(root=dataset_root, split="train", transforms=transform)
    dataset_test = StoneDataset(root=dataset_root, split="test", transforms=transform)

    print(f"Train size: {len(dataset_train)}")
    print(f"Test size: {len(dataset_test)}")

    image, label = dataset_train[0]
    print(f"Sample image shape: {image.shape}, Label: {label}")