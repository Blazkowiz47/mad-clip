import argparse
import json
from typing import Dict, Tuple
from inference import main as inference
from morph_desc import morph_list, genuine_list
import numpy as np


def main() -> None:
    results: Dict[str, Tuple[float, float, float, float]] = {}
    for id, (gdesc, mdesc) in enumerate(zip(genuine_list, morph_list)):
        printers = ["digital", "DNP", "rico"]
        morphs = [
            "lma    ",
            "lmaubo ",
            "mipgan2",
            "mordiff",
            "pipe   ",
        ]
        for printer in printers:
            for morph in morphs:
                args = argparse.Namespace(
                    morph=morph.strip(),
                    printer=printer,
                    model="./checkpoints/best_cross_domain_loss_model.pt",
                    mdesc=mdesc,
                    gdesc=gdesc,
                    descid=id,
                )
                (
                    eer,
                    bon_correct,
                    mor_correct,
                    bon_incorrect,
                    mor_incorrect,
                    gen,
                    mor,
                ) = inference(args)  # noqa: E501
                np.save(
                    f"scores/fine_tune_p{id}_{printer}_{morph.strip()}_gen.npy", gen
                )
                np.save(
                    f"scores/fine_tune_p{id}_{printer}_{morph.strip()}_mor.npy", mor
                )
                results[f"{id}_{printer}_{morph}"] = (
                    eer,
                    bon_correct * 100 / (bon_correct + bon_incorrect),
                    mor_correct * 100 / (mor_correct + mor_incorrect),
                    (bon_correct + mor_correct)
                    * 100
                    / (bon_correct + bon_incorrect + mor_correct + mor_incorrect),
                )
        print("Bonafide prompt:", gdesc)
        print("Morph prompt:", mdesc)
        for printer in printers:
            print(
                printer,
                ":    ",
                *["eer", "bon accuracy", "mor accuracy", "total accuracy"],
            )
            for morph in morphs:
                print(
                    "\t",
                    morph,
                    *[round(x, 2) for x in results[f"{id}_{printer}_{morph}"]],
                )
    with open("fine_tune_results.json", "w+") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    main()
