import argparse
from inference import main as inference
from metrics import get_bpcer
import numpy as np


def exp1(morphs, printers, model) -> None:
    res = f"\nExperiment 1 ({model})\n"
    pref = "vit32_" if "32" in model else ""
    for mprinter in printers:
        res += "Trained on: " + mprinter
        res += "\n\t"
        for printer in printers:
            res += printer + "\t"
        res += "\n"

        for morph in morphs:
            res += morph + ": "
            morph = morph.strip()
            for printer in printers:
                args = argparse.Namespace(
                    morph=morph.strip(),
                    printer=printer,
                    model=f"./checkpoints/{pref}exp1_{morph}_{mprinter}.pt",
                    backbone=model,
                )
                eer, _, _, _, _, gen, mor = inference(args)
                bpcer = get_bpcer(gen, mor, 10)
                res += str(round(bpcer, 2)) + "\t"
                np.save(
                    f"./scores/{pref}exp1_{morph}_{mprinter}_{printer}_gen.npy", gen
                )
                np.save(
                    f"./scores/{pref}exp1_{morph}_{mprinter}_{printer}_mor.npy", mor
                )
            res += "\n"

    print(res)


def exp2(morphs, printers, model) -> None:
    #     for printer in printers:
    #         args = argparse.Namespace(
    #             printer=printer,
    #             morph=",".join(morphs),
    #             model_name=f"vit32_exp2_{printer}",
    #         )

    res = f"\nExperiment 2 ({model})\n"
    pref = "vit32_" if "32" in model else ""
    for mprinter in printers:
        res += "Trained on: " + mprinter
        res += "\n\t"
        for printer in printers:
            res += printer + "\t"
        res += "\n"

        for morph in morphs:
            res += morph + ": "
            morph = morph.strip()
            for printer in printers:
                args = argparse.Namespace(
                    morph=morph.strip(),
                    printer=printer,
                    model=f"./checkpoints/{pref}exp2_{mprinter}.pt",
                    backbone=model,
                )
                eer, _, _, _, _, gen, mor = inference(args)
                bpcer = get_bpcer(gen, mor, 10)
                res += str(round(bpcer, 2)) + "\t"
                np.save(
                    f"./scores/{pref}exp2_{morph}_{mprinter}_{printer}_gen.npy", gen
                )
                np.save(
                    f"./scores/{pref}exp2_{morph}_{mprinter}_{printer}_mor.npy", mor
                )
            res += "\n"
    print(res)


def exp3(morphs, printers, model) -> None:
    #     for morph in morphs:
    #         args = argparse.Namespace(
    #             printer=",".join(printers),
    #             morph=",".join([m for m in morphs if m != morph]),
    #             model_name=f"vit32_exp3_{morph}",
    #         )
    res = f"\nExperiment 3 ({model})\n"
    pref = "vit32_" if "32" in model else ""
    for mmorph in morphs:
        res += "Trained on: " + mmorph.strip()
        mmorph = mmorph.strip()
        res += "\n\t"
        for printer in printers:
            res += printer + "\t"
        res += "\n"

        for morph in morphs:
            res += morph + ": "
            morph = morph.strip()
            for printer in printers:
                args = argparse.Namespace(
                    morph=morph.strip(),
                    printer=printer,
                    model=f"./checkpoints/{pref}exp3_{mmorph}.pt",
                    backbone=model,
                )
                eer, _, _, _, _, gen, mor = inference(args)
                bpcer = get_bpcer(gen, mor, 10)
                res += str(round(bpcer, 2)) + "\t"
                np.save(f"./scores/{pref}exp3_{morph}_{mmorph}_{printer}_gen.npy", gen)
                np.save(f"./scores/{pref}exp3_{morph}_{mmorph}_{printer}_mor.npy", mor)
            res += "\n"
    print(res)


def main() -> None:
    morphs = [
        "lma    ",
        "lmaubo ",
        "mipgan2",
        "mordiff",
        "pipe   ",
    ]
    printers = ["digital", "DNP"]

    model = "vit-large-patch14"
    exp1(morphs, printers, model)
    exp2(morphs, printers, model)
    exp3(morphs, printers, model)

    model = "vit-base-patch32"
    exp1(morphs, printers, model)
    exp2(morphs, printers, model)
    exp3(morphs, printers, model)


if __name__ == "__main__":
    main()
