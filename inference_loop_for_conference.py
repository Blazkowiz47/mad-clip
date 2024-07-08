import argparse
import json
from typing import Dict, Tuple
from inference import main as inference
from metrics import get_bpcer
from morph_desc import morph_list, genuine_list
import numpy as np


def exp1(morphs, printers) -> None:
    res = "\nExperiment 1 (vit base 32)\n"
    for mprinter in printers:
        res += "Trained on: " + mprinter
        res += "\n\t"
        for printer in printers:
            res += printer + "\t"
        res += "\n"

        for morph in morphs:
            res += morph + ": "
            for printer in printers:
                args = argparse.Namespace(
                    morph=morph.strip(),
                    printer=printer,
                    model=f"./checkpoints/vit32_exp1_{morph}_{mprinter}",
                )
                eer, _, _, _, _, gen, mor = inference(args)
                bpcer = get_bpcer(gen, mor, 10)
                res += str(round(bpcer, 2)) + "\t"
                np.save(
                    f"./scores/vit32_exp1_{morph}_{mprinter}_{printer}_gen.npy", gen
                )
                np.save(
                    f"./scores/vit32_exp1_{morph}_{mprinter}_{printer}_mor.npy", mor
                )
            res += "\n"


def exp2(morphs, printers) -> None:
    for printer in printers:
        args = argparse.Namespace(
            printer=printer,
            morph=",".join(morphs),
            model_name=f"vit32_exp2_{printer}",
        )


def exp3(morphs, printers) -> None:
    for morph in morphs:
        args = argparse.Namespace(
            printer=",".join(printers),
            morph=",".join([m for m in morphs if m != morph]),
            model_name=f"vit32_exp3_{morph}",
        )


def main() -> None:
    results: Dict[str, Tuple[float, float, float, float]] = {}
    morphs = [
        "lma    ",
        "lmaubo ",
        "mipgan2",
        "mordiff",
        "pipe   ",
    ]
    for tmorph in morphs:
        tmorph = tmorph.strip()
        for id, (gdesc, mdesc) in enumerate(zip(genuine_list, morph_list)):
            printers = ["digital", "DNP", "rico"]
            for printer in printers:
                for morph in morphs:
                    args = argparse.Namespace(
                        morph=morph.strip(),
                        printer=printer,
                        model=f"./checkpoints/best_cross_domain_{tmorph}_accuracy_model.pt",
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
                    ) = inference(
                        args
                    )  # noqa: E501
                    np.save(
                        f"scores/fine_tune_on_{tmorph}_p{id}_{printer}_{morph.strip()}_gen.npy",
                        gen,
                    )
                    np.save(
                        f"scores/fine_tune_on_{tmorph}_p{id}_{printer}_{morph.strip()}_mor.npy",
                        mor,
                    )
                    results[f"{tmorph}_{id}_{printer}_{morph}"] = (
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
                    tmorph,
                    printer,
                    ":    ",
                    *["eer", "bon accuracy", "mor accuracy", "total accuracy"],
                )
                for morph in morphs:
                    print(
                        "\t",
                        morph,
                        *[
                            round(x, 2)
                            for x in results[f"{tmorph}_{id}_{printer}_{morph}"]
                        ],
                    )

    with open("fine_tune_results.json", "w+") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    main()
