import argparse
import json
from typing import Dict, Tuple
from inference import main as inference
from evaluation import main as eval
from morph_desc import morph_list, genuine_list
import numpy as np


def main() -> None:
    results: Dict[str, Tuple[float]] = {}
    morphs = [
        "lma    ",
        "lmaubo ",
        "mipgan2",
        "mordiff",
        "pipe   ",
    ]
    for tmorph in ["resnet50", "resnet101", "vgg19", "vitb", "vitl"]:
        tmorph = tmorph.strip()
        printers = ["digital", "DNP", "rico"]
        for printer in printers:
            for morph in morphs:
                args = argparse.Namespace(
                    morph=morph.strip(),
                    printer=printer,
                    model=tmorph,
                )
                eer, gen, mor = eval(args)
                np.save(
                    f"scores/eval_{tmorph}_{printer}_{morph.strip()}_gen.npy",
                    gen,
                )
                np.save(
                    f"scores/eval_{tmorph}_{printer}_{morph.strip()}_mor.npy",
                    mor,
                )
                results[f"{tmorph}_{printer}_{morph}"] = (eer,)
                #                         bon_correct * 100 / (bon_correct + bon_incorrect),
                #                         mor_correct * 100 / (mor_correct + mor_incorrect),
                #                         (bon_correct + mor_correct)
                #                         * 100
                #                         / (bon_correct + bon_incorrect + mor_correct + mor_incorrect),
        #             print("Bonafide prompt:", gdesc)
        #             print("Morph prompt:", mdesc)
        for printer in printers:
            print(
                tmorph,
                printer,
                ":    ",
                *["eer", "bon accuracy", "mor accuracy", "total accuracy"],
            )
            for morph in morphs:
                print(
                    "\t",
                    morph,
                    *[round(x, 2) for x in results[f"{tmorph}_{printer}_{morph}"]],
                )

    with open("backbone.json", "w+") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    main()
