import argparse
from typing import List, Tuple

from transformers import CLIPModel, CLIPProcessor
import torch

from dataset import Wrapper
from tqdm import tqdm
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

parser.add_argument(
    "--mdesc",
    type=str,
    default="",
)

parser.add_argument(
    "--gdesc",
    type=str,
    default="",
)

with open("./bonafide_prompt.txt") as fp:
    bonafide_prompt = fp.read()

with open("./morph_prompt.txt") as fp:
    morph_prompt = fp.read()


def classify(
    imgs, lbls, processor, model
) -> Tuple[int, int, int, int, List[float], List[float]]:
    inputs = processor(
        text=[
            morph_prompt,
            bonafide_prompt,
        ],
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
    lbls = lbls.argmax(dim=1)
    preds = probs.argmax(dim=1)

    boncorrect = 0
    morcorrect = 0
    bonincorrect = 0
    morincorrect = 0
    morphscores: List[float] = []
    genscoresscores: List[float] = []
    for lbl, pred, prob in zip(lbls, preds, probs):
        if lbl == 1.0:
            if pred == lbl:
                boncorrect += 1
            else:
                bonincorrect += 1
            genscoresscores.append(prob[1].detach().cpu().item())
        else:
            if pred == lbl:
                morcorrect += 1
            else:
                morincorrect += 1
            morphscores.append(prob[1].detach().cpu().item())

    return (
        boncorrect,
        bonincorrect,
        morcorrect,
        morincorrect,
        genscoresscores,
        morphscores,
    )


def main(
    args: argparse.Namespace,
) -> Tuple[float, int, int, int, int, List[float], List[float]]:
    try:
        global bonafide_prompt, morph_prompt
        if args.gdesc:
            bonafide_prompt = args.gdesc
        if args.mdesc:
            morph_prompt = args.mdesc
        rdir = (
            "/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/PRINT_SCAN/"  # noqa: E501
        )
        printer = args.printer
        morph_type = args.morph
        wrapper = Wrapper(rdir, morph_type, printer, 16)
        testds = wrapper.get_test()

        model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        if args.model:
            model.load_state_dict(torch.load(args.model))

        model = model.cuda()
        model.eval()
        bon_correct = 0
        bon_incorrect = 0
        mor_correct = 0
        mor_incorrect = 0
        genscores: List[float] = []
        impscores: List[float] = []
        #     for imgs, lbls in tqdm(testds):
        for imgs, lbls in tqdm(testds):
            imgs, lbls = imgs, lbls.cuda()
            bcorrect, bincorrect, mcorrect, mincorrect, gs, ms = classify(
                imgs, lbls, processor, model
            )
            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            genscores.extend(gs)
            impscores.extend(ms)

        eer, *_ = get_eer(genscores, impscores)
        print(f"{printer} {morph_type}")
        print(
            f"Bonafide ({bon_correct + bon_incorrect}): correct: {bon_correct} incorrect: {bon_incorrect}"  # noqa: E501
        )
        print(
            f"Morph ({mor_correct + mor_incorrect}): correct: {mor_correct} incorrect: {mor_incorrect}"  # noqa: E501
        )
        print(f"{printer} {morph_type}")

        return (
            eer,
            bon_correct,
            mor_correct,
            bon_incorrect,
            mor_incorrect,
            genscores,
            impscores,
        )
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
