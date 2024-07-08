import argparse
from train import main as train


def exp1(morphs, printers) -> None:
    for printer in printers:
        for morph in morphs:
            args = argparse.Namespace(
                printer=printer,
                morph=",".join([m for m in morphs if m != morph]),
                model_name=f"vit32_exp1_{morph}_{printer}",
            )
            train(args)


def exp2(morphs, printers) -> None:
    for printer in printers:
        args = argparse.Namespace(
            printer=printer,
            morph=",".join(morphs),
            model_name=f"vit32_exp2_{printer}",
        )
        train(args)


def exp3(morphs, printers) -> None:
    for morph in morphs:
        args = argparse.Namespace(
            printer=",".join(printers),
            morph=",".join([m for m in morphs if m != morph]),
            model_name=f"vit32_exp3_{morph}",
        )
        train(args)


def main() -> None:
    morphs = [
        "lma",
        "lmaubo",
        "mipgan2",
        "mordiff",
        "pipe",
    ]
    printers = ["digital", "DNP", "rico"]

    exp1(morphs, printers)
    exp2(morphs, printers)
    exp3(morphs, printers)


if __name__ == "__main__":
    main()
