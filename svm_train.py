from typing import Tuple
from numpy.typing import NDArray
from sklearn import svm


def get_results(
    trainds: Tuple[NDArray, NDArray], testds: Tuple[NDArray, NDArray]
) -> None:
    clf = svm.SVC()
    clf.fit(trainds[0], trainds[1].argmax(axis=1))
    probs = clf.decision_function(testds[0])
    print(probs.shape)


def evaluate(
    printer: str,
    morph_type: str,
    descid: str,
) -> None: ...
    
    train_fname = f"./embeddings/imgfeat_test_{descid}_{printer}_{morph_type}.npy",
    test_fname = f"./embeddings/imgfeat_test_{descid}_{printer}_{morph_type}.npy",
