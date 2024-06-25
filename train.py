import argparse
import os
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import (
    CrossEntropyLoss,
    Module,
)
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel

from morph_desc import morph_list, genuine_list
from dataset import Wrapper

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

with open("./bonafide_prompt.txt") as fp:
    bonafide_prompt = fp.read()

with open("./morph_prompt.txt") as fp:
    morph_prompt = fp.read()


def train_classifier(
    imgs: torch.Tensor,
    lbls: torch.Tensor,
    text: List[str],
    processor: CLIPProcessor,
    model: CLIPModel,
    loss_fn: Module,
) -> Tuple[torch.Tensor, int, int, int, int]:
    inputs = processor(
        text=text,
        images=imgs,
        return_tensors="pt",
        padding=True,
    )
    # print(*[f"{k}: {v.shape}" for k, v in inputs.items()])
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    loss = loss_fn(probs, lbls)
    lbls = lbls.argmax(dim=1)
    probs = probs.argmax(dim=1)

    boncorrect = 0
    morcorrect = 0
    bonincorrect = 0
    morincorrect = 0
    for lbl, prob in zip(lbls, probs):
        if lbl == 1.0:
            if prob == lbl:
                boncorrect += 1
            else:
                bonincorrect += 1
        else:
            if prob == lbl:
                morcorrect += 1
            else:
                morincorrect += 1

    return loss, boncorrect, bonincorrect, morcorrect, morincorrect


def main(args: argparse.Namespace) -> None:
    rdir = "/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/PRINT_SCAN/"  # noqa: E501
    printer = args.printer
    morph_type = args.morph

    if "," in printer:
        printer = printer.split(",")
    if "," in morph_type:
        morph_type = morph_type.split(",")

    wrapper = Wrapper(rdir, morph_type, printer, 32)
    trainds = wrapper.get_train()
    testds = wrapper.get_test()

    model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    for params in model.parameters():
        params.requires_grad = False

    for p in model.text_projection.parameters():
        p.requires_grad = True

    for p in model.visual_projection.parameters():
        p.requires_grad = True

    # vplayer = Sequential(
    # BatchNorm1d(model.projection_dim),
    # LeakyReLU(),
    # Linear(model.projection_dim, model.projection_dim),
    # )
    # for params in vplayer.parameters():
    # params.requires_grad = True

    model.logit_scale.requires_grad = True

    # aplayer = Sequential(
    # BatchNorm1d(model.projection_dim),
    # LeakyReLU(),
    # Linear(model.projection_dim, model.projection_dim),
    # )
    # for params in aplayer.parameters():
    # params.requires_grad = True
    #
    # model.visual_projection = Sequential(
    # model.visual_projection,
    # vplayer,
    # )
    # model.text_projection = Sequential(
    # model.text_projection,
    # aplayer,
    # )

    model = model.cuda()
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        1e-5,
    )
    loss_fn = CrossEntropyLoss().cuda()
    best_train_loss: float = float("inf")
    best_test_accuracy: float = 0.0
    for epoch in range(10):
        model.train()
        bon_correct = 0
        bon_incorrect = 0
        mor_correct = 0
        mor_incorrect = 0
        total_loss = 0.0
        print("Epoch:", epoch)
        for imgs, lbls in tqdm(trainds, desc="Training"):
            optimizer.zero_grad()
            imgs, lbls = imgs, lbls.cuda()
            loss, bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(
                imgs,
                lbls,
                [random.choice(morph_list), random.choice(genuine_list)],
                processor,
                model,
                loss_fn,
            )
            loss.backward()
            optimizer.step()
            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            total_loss += loss.detach().cpu().item()

        print("Train Loss:", total_loss)
        if best_train_loss > total_loss:
            os.makedirs("checkpoints", exist_ok=True)
            best_train_loss = total_loss
        #             torch.save(
        #                 model.state_dict(),
        #                 f"./checkpoints/best_cross_domain_{morph_type}_loss_model.pt",
        #             )

        print(
            f"Bonafide ({bon_correct + bon_incorrect}): correct: {bon_correct} incorrect: {bon_incorrect}"  # noqa: E501
        )
        print(
            f"Morph ({mor_correct + mor_incorrect}): correct: {mor_correct} incorrect: {mor_incorrect}"  # noqa: E501
        )

        model.eval()
        bon_correct = 0
        bon_incorrect = 0
        mor_correct = 0
        mor_incorrect = 0
        total_loss = 0.0
        for imgs, lbls in tqdm(testds):
            imgs, lbls = imgs, lbls.cuda()
            loss, bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(
                imgs,
                lbls,
                [random.choice(morph_list), random.choice(genuine_list)],
                processor,
                model,
                loss_fn,
            )
            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            total_loss += loss.detach().cpu().item()

        print("Test Loss:", total_loss)
        print(
            f"Bonafide ({bon_correct + bon_incorrect}): correct: {bon_correct} incorrect: {bon_incorrect}"  # noqa: E501
        )
        print(
            f"Morph ({mor_correct + mor_incorrect}): correct: {mor_correct} incorrect: {mor_incorrect}"  # noqa: E501
        )

        accuracy = (bon_correct + mor_correct) / (bon_incorrect + mor_incorrect)
        if accuracy > best_test_accuracy:
            best_test_accuracy = accuracy
            torch.save(
                model.state_dict(),
                f"./checkpoints/best_cross_domain_{morph_type}_accuracy_model.pt",
            )


def set_seeds(seed: int = 2024):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
