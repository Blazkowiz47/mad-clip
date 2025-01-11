from typing import Dict, Tuple
import numpy as np
from metrics import get_bpcer


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
                gen = np.load(
                    f"scores/eval_{tmorph}_{printer}_{morph.strip()}_gen.npy",
                )
                mor = np.load(
                    f"scores/eval_{tmorph}_{printer}_{morph.strip()}_mor.npy",
                )
                bpcer = get_bpcer(gen, mor, 10)
                results[f"{tmorph}_{printer}_{morph}"] = (bpcer,)

    for tmorph in ["resnet50", "resnet101", "vgg19", "vitb", "vitl"]:
        printers = ["digital", "DNP", "rico"]

        result = tmorph + "\n\t" + "\t".join(printers)
        result += "\n"

        for morph in morphs:
            result += morph + "  "
            for printer in printers:
                result += (
                    str(round(results[f"{tmorph}_{printer}_{morph}"][0], 2)) + "\t"
                )
            result += "\n"

        print(result)


if __name__ == "__main__":
    main()
