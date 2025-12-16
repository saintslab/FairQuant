from typing import Tuple, List, Optional, Union
import os
import csv
import random
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
import gdown
import zipfile
import logging
import tarfile

def download_and_extract(url, root_dir, dataset_name):
    os.makedirs(root_dir, exist_ok=True)

    if url.endswith(".tar"):
        file_path = os.path.join(root_dir, f"{dataset_name}.tar")
        is_tar = True
    else:
        file_path = os.path.join(root_dir, f"{dataset_name}.zip")
        is_tar = False

    logging.info(f"{dataset_name} not found at '{root_dir}'. Starting automatic download...")
    logging.info(f"Downloading from: {url}")
    logging.info(f"Saving to: {file_path}")
    logging.info("This may take a significant amount of time and disk space.")

    gdown.download(url, file_path, quiet=False)

    logging.info(f"Download complete. Extracting '{file_path}'...")

    if is_tar:
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(path=root_dir)
    else:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)

    os.remove(file_path)

    logging.info(f"Extraction complete. {dataset_name} is ready.")



class ImageTupleDataset(Dataset):
    def __init__(self, items: List[tuple], transform):
        self.items, self.transform = items, transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        p, y, g = self.items[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, int(y), int(g)

class CelebAWrapper(Dataset):
    def __init__(self, base_dataset, target_attr_idx, sensitive_attr_idx):
        self.base = base_dataset
        self.target_attr_idx = target_attr_idx
        self.sensitive_attr_idx = sensitive_attr_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, attrs = self.base[idx]
        target = attrs[self.target_attr_idx]
        sensitive = attrs[self.sensitive_attr_idx]
        return img, int(target), int(sensitive)

def _apply_subset(ds: Dataset, subset: Optional[Union[float, int]]):
    if subset is None: return ds
    n = len(ds)
    k = max(1, int(n * subset)) if isinstance(subset, float) else min(n, int(subset))
    return Subset(ds, torch.randperm(n)[:k].tolist())



def _celeba_loaders(data_root: str, batch_size: int, image_size: int, num_workers: int, train_subset, test_subset, pin_memory: bool, target_attribute: str, sensitive_attribute: str):
    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15), # A sensible default for rotation
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.CenterCrop(image_size), 
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = T.Compose([
        T.Resize(image_size), 
        T.CenterCrop(image_size), 
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_base = torchvision.datasets.CelebA(root=data_root, split="train", target_type="attr", transform=train_transform, download=True)
    test_base = torchvision.datasets.CelebA(root=data_root, split="test", target_type="attr", transform=test_transform, download=True)

    try:
        target_attr_idx = train_base.attr_names.index(target_attribute)
        sensitive_attr_idx = train_base.attr_names.index(sensitive_attribute)
    except ValueError as e:
        raise ValueError(f"{e}. Please choose from the available CelebA attributes: {train_base.attr_names}")

    logging.info(f"Using CelebA dataset. Target: '{target_attribute}', Sensitive: '{sensitive_attribute}'")
    
    train_ds = CelebAWrapper(train_base, target_attr_idx, sensitive_attr_idx)
    test_ds = CelebAWrapper(test_base, target_attr_idx, sensitive_attr_idx)
    
    train_ds = _apply_subset(train_ds, train_subset)
    test_ds = _apply_subset(test_ds, test_subset)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, test_loader, 2, [f"Not {target_attribute}", target_attribute], 2, [f"Not {sensitive_attribute}", sensitive_attribute]


def _fitzpatrick17k_loaders(data_root: str, batch_size: int, image_size: int, num_workers: int, train_subset, test_subset, pin_memory: bool, binary_grouping: bool):
    base = os.path.join(data_root, "Fitzpatrick17k")
    if not os.path.isdir(base):
        url = "https://notredame.box.com/shared/static/pjf9kw5y1rtljnh81kuri4poecuiqngf.tar"
        download_and_extract(url, data_root, "Fitzpatrick17k")

    images_dir = os.path.join(base, "images")
    main_meta_csv = os.path.join(base, "fitzpatrick17k.csv")
    train_hashes_csv = os.path.join(base, "train_score.csv") 
    val_hashes_csv = os.path.join(base, "all_val.csv")

    if not all(os.path.isfile(f) for f in [main_meta_csv, train_hashes_csv, val_hashes_csv]):
        raise RuntimeError(f"Fitzpatrick17k data not found or corrupted in {base}. Missing a required CSV file.")

    metadata = {}
    for r in csv.DictReader(open(main_meta_csv, "r")):
        if r.get("md5hash"):
            metadata[r["md5hash"]] = r

    def get_hashes_from_split_file(csv_path):
        hashes = set()
        for r in csv.DictReader(open(csv_path, "r")):
            if r.get("md5hash"):
                hashes.add(r["md5hash"])
        return hashes

    train_hashes = get_hashes_from_split_file(train_hashes_csv)
    val_hashes = get_hashes_from_split_file(val_hashes_csv)

    all_rows = metadata.values()
    all_rows = [r for r in all_rows if str(r.get("fitzpatrick", "UNK")) not in ["UNK", "-1"]]
    labels, label_to_idx = sorted(list({r["label"] for r in all_rows})), {c: i for i, c in enumerate(sorted(list({r["label"] for r in all_rows})))}

    if binary_grouping:
        logging.info("Using binary grouping for Fitzpatrick17k (Light vs. Dark skin).")
        group_names = ["Light Skin (1-3)", "Dark Skin (4-6)"]
        binary_map = {'1': 0, '2': 0, '3': 0, '4': 1, '5': 1, '6': 1}
        group_to_idx_func = lambda g_name: binary_map.get(str(g_name), 0)
    else:
        logging.info("Using multi-class grouping for Fitzpatrick17k.")
        raw_group_values = set(str(r.get("fitzpatrick")) for r in all_rows if r.get("fitzpatrick"))
        group_names = sorted(list(raw_group_values))
        group_to_idx = {name: i for i, name in enumerate(group_names)}
        group_to_idx_func = lambda g_name: group_to_idx.get(g_name)

    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = T.Compose([
        T.Resize((image_size, image_size)), T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def build_items(target_hashes: set):
        items = []
        for h in target_hashes:
            if h in metadata:
                r = metadata[h]
                filename = f"{h}.jpg"
                filepath = os.path.join(images_dir, filename)
                if not os.path.isfile(filepath): continue
                
                fitz_val = str(r.get("fitzpatrick", "UNK"))
                if fitz_val in ["UNK", "-1"]: continue
                
                y = label_to_idx[r["label"]]
                g = group_to_idx_func(fitz_val)
                items.append((filepath, y, g))
        
        if not items:
            raise ValueError(f"FATAL: Created an empty dataset. No matching images found for the target hashes.")
        return items

    train_ds = ImageTupleDataset(build_items(train_hashes), train_transform)
    test_ds = ImageTupleDataset(build_items(val_hashes), test_transform)
    
    train_ds, test_ds = _apply_subset(train_ds, train_subset), _apply_subset(test_ds, test_subset)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader, len(labels), labels, len(group_names), group_names


def _isic2019_loaders(data_root: str, batch_size: int, image_size: int, num_workers: int, train_subset, test_subset, pin_memory: bool):
    base = os.path.join(data_root, "ISIC2019_train")
    
    if not os.path.isdir(os.path.join(base, "ISIC_2019_Training_Input")):
        url = "https://notredame.box.com/shared/static/uw8g5urs7m4n4ztxfo100kkga6arzi9k.tar"
        download_and_extract(url, data_root, "ISIC2019_train")

    in_dir, gt_csv, meta_csv = os.path.join(base, "ISIC_2019_Training_Input"), os.path.join(base, "ISIC_2019_Training_GroundTruth.csv"), os.path.join(base, "ISIC_2019_Training_Metadata.csv")
    if not (os.path.isdir(in_dir) and os.path.isfile(gt_csv)): raise RuntimeError(f"ISIC2019 data not found or corrupted in {base}.")
    
    gt_rows = [r for r in csv.DictReader(open(gt_csv, "r"))]
    class_names = [c for c in gt_rows[0].keys() if c != "image"]
    label_for_row = lambda r: int(max(range(len(class_names)), key=lambda i: float(r[class_names[i]])))
    
    group_names, group_to_idx = ["UNK", "female", "male"], {"UNK":0, "female":1, "male":2}
    sex_map = {}
    if os.path.isfile(meta_csv):
        for r in csv.DictReader(open(meta_csv, "r")):
            sex = r.get("sex", "UNK").lower()
            sex_map[r.get("image")] = group_to_idx.get(sex if sex in group_names else "UNK")

    all_items = [ (os.path.join(in_dir, r["image"] + ".jpg"), label_for_row(r), sex_map.get(r["image"], group_to_idx["UNK"])) for r in gt_rows if os.path.isfile(os.path.join(in_dir, r["image"] + ".jpg"))]
    random.Random(42).shuffle(all_items)
    
    train_cut = int(0.6 * len(all_items))
    val_cut = int(0.8 * len(all_items))
    train_items, test_items = all_items[:train_cut], all_items[train_cut:val_cut]
    
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((image_size, image_size)), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds, test_ds = ImageTupleDataset(train_items, train_transform), ImageTupleDataset(test_items, test_transform)
    train_ds, test_ds = _apply_subset(train_ds, train_subset), _apply_subset(test_ds, test_subset)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader, len(class_names), class_names, len(group_names), group_names

def get_dataloaders(
    dataset: str = "celeba", data_root: str = "./data", batch_size: int = 128,
    image_size: int = 224, num_workers: int = 2, train_subset: Optional[Union[float, int]] = None,
    test_subset: Optional[Union[float, int]] = None,
    target_attribute: str = "Blond_Hair",
    sensitive_attribute: str = "Male",
    fitzpatrick_binary_grouping: bool = False
):
    os.makedirs(data_root, exist_ok=True)
    pin_memory = torch.cuda.is_available()
    dataset = dataset.lower()
    
    if dataset == "celeba":
        return _celeba_loaders(data_root, batch_size, image_size, num_workers, train_subset, test_subset, pin_memory, target_attribute, sensitive_attribute)
    elif dataset == "fitzpatrick17k":
        return _fitzpatrick17k_loaders(data_root, batch_size, image_size, num_workers, train_subset, test_subset, pin_memory, fitzpatrick_binary_grouping)
    elif dataset in ("isic2019", "isic-2019"):
        return _isic2019_loaders(data_root, batch_size, image_size, num_workers, train_subset, test_subset, pin_memory)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choices: celeba, fitzpatrick17k, isic2019")