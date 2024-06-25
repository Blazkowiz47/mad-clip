import os
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class Wrapper:
    def __init__(
        self,
        rdir: str,
        morphtype: List[str] | str,
        printer: List[str] | str,
        batch_size: int = 2,
    ):
        if isinstance(printer, str):
            printer = [printer]
        if isinstance(morphtype, str):
            morphtype = [morphtype]

        self.bondir: List[str] = []
        self.mordir: List[str] = []
        for p in printer:
            self.bondir.append(os.path.join(rdir, p, "bonafide"))
            self.mordir.extend([os.path.join(rdir, p, "morph", m) for m in morphtype])

            self.batch_size = batch_size

    def loop_splitset(self, ssplit: str) -> DataLoader:
        mordirs = []
        bondirs = []
        for mordir in self.mordir:
            tmordir = os.path.join(mordir, ssplit)
            if os.path.isdir(os.path.join(tmordir, "FaceDetect")):
                tmordir = os.path.join(tmordir, "FaceDetect")
            mordirs.append(tmordir)

        for bondir in self.bondir:
            bondir = os.path.join(bondir, ssplit)
            if os.path.isdir(os.path.join(bondir, "FaceDetect")):
                bondir = os.path.join(bondir, "FaceDetect")
            bondirs.append(bondir)

        data: List[Tuple[str, int]] = []

        for bondir in bondirs:
            for fname in os.listdir(bondir):
                if fname.lower().endswith(".jpg") or fname.lower().endswith(".png"):
                    data.append((os.path.join(bondir, fname), 1))

        for mordir in mordirs:
            for fname in os.listdir(mordir):
                if fname.lower().endswith(".jpg") or fname.lower().endswith(".png"):
                    data.append((os.path.join(mordir, fname), 0))

        return DataLoader(
            DatasetGenerator(data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def get_test(self) -> DataLoader:
        return self.loop_splitset("test")

    def get_train(self) -> DataLoader:
        return self.loop_splitset("train")


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Tuple[str, int]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        fname, lbl = self.data[index]
        img = self.transform(fname)
        label = [0, 0]
        label[lbl] = 1
        return torch.tensor(img).float(), torch.tensor(label).float()

    def transform(self, fname: str) -> NDArray:
        img = Image.open(fname)
        imgarray = np.array(img)
        imgarray = cv2.resize(imgarray, [224, 224])
        return imgarray
