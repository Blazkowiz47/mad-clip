import argparse
from typing import List, Tuple

import torch
from torch.nn import Module
from torch.nn.functional import cosine_similarity
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    VGG19_Weights,
    ViT_B_32_Weights,
    ViT_L_32_Weights,
    ViT_H_14_Weights,
    resnet50,
    resnet101,
    vgg19,
    vit_b_32,
    vit_h_14,
    vit_l_32,
)
from tqdm import tqdm

from dataset import Wrapper
from metrics import get_eer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--printer",
    type=str,
    default="digital",
)

parser.add_argument(
    "-m",
    "--morph",
    type=str,
    default="lma",
)

parser.add_argument(
    "-o",
    "--model",
    type=str,
    default="",
)


def classify(enrollfeat: torch.Tensor, probefeat: torch.Tensor) -> torch.Tensor:
    return cosine_similarity(enrollfeat, probefeat, dim=1).max().detach().cpu()
    return (enrollfeat - probefeat).square().sum(dim=1).min().detach().cpu()


class CustomModel(Module):
    def __init__(self, model: str) -> None:
        super(CustomModel, self).__init__()
        self.backbone = self.get_model(model)
        self.model = model

    def get_model(self, model: str) -> Module:
        if model == "resnet50":
            return resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()
        if model == "resnet101":
            return resnet101(weights=ResNet101_Weights.DEFAULT).eval().cuda()
        if model == "vgg19":
            return vgg19(weights=VGG19_Weights.DEFAULT).eval().cuda()
        if model == "vitb":
            return vit_b_32(weights=ViT_B_32_Weights.DEFAULT).eval().cuda()
        if model == "vitl":
            return vit_l_32(weights=ViT_L_32_Weights.DEFAULT).eval().cuda()
        if model == "vith":
            return vit_h_14(weights=ViT_H_14_Weights.DEFAULT).eval().cuda()
        raise NotImplementedError()

    def vgg_forward(self, x: torch.Tensor):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def vit_forward(self, x: torch.Tensor):
        x = self.backbone._process_input(x)
        n = x.shape[0]
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)
        x = x[:, 0]
        return x

    def resnet_forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = x.transpose(2, 3).transpose(1, 2)
        if "resnet" in self.model:
            return self.resnet_forward(x)
        if "vgg" in self.model:
            return self.vgg_forward(x)
        if "vit" in self.model:
            return self.vit_forward(x)
        raise NotImplementedError()


def main(
    args: argparse.Namespace,
) -> Tuple[float, List[float], List[float]]:
    try:
        rdir = (
            "ROOT_DIR"  # noqa: E501
        )
        printer = args.printer
        morph_type = args.morph
        wrapper = Wrapper(rdir, morph_type, printer, 128)
        trainds = wrapper.get_train(x=0)
        testds = wrapper.get_test(batch_size=1)

        model = CustomModel(args.model).eval().cuda()

        model = model.cuda()
        model.eval()
        genscores: List[float] = []
        impscores: List[float] = []

        probefeatures = []
        for probe, _ in tqdm(trainds):
            probe_features = model(probe.float().cuda())
            probefeatures.append(probe_features.detach().cpu())

        for enroll, enrolbl in tqdm(testds):
            enroll_features = model(enroll.float().cuda())
            #             enroll_features = torch.cat([enroll_features] * 64, dim=0)
            enrolbl = enrolbl.argmax(dim=1).cpu().item()
            mgs: torch.Tensor | None = None
            mms: torch.Tensor | None = None

            for probe_features in probefeatures:
                #                 probe_features = model(probe.float().cuda())
                score = classify(enroll_features, probe_features.cuda())
                if enrolbl:
                    if mgs:
                        mgs = torch.max(mgs, score)
                    else:
                        mgs = score
                else:
                    if mms:
                        mms = torch.max(mms, score)
                    else:
                        mms = score

            if mgs:
                genscores.append(mgs.item())
            if mms:
                impscores.append(mms.item())
        impscores = sorted(impscores)
        genscores = sorted(genscores)
        print(impscores[:40])
        print(genscores[:40])
        eer, *_ = get_eer(genscores, impscores)
        print(f"{printer} {morph_type}:", eer)

        return (
            eer,
            genscores,
            impscores,
        )

    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
