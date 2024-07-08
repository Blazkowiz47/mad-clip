import numpy as np


def get_bpcer(genuine, imposter, apcer, bins=10_001):
    genuine = np.squeeze(np.array(genuine))
    imposter = np.squeeze(np.array(imposter))
    far = np.ones(bins)
    frr = np.ones(bins)
    mi = np.min(imposter)
    mx = np.max(genuine)
    thresholds = np.linspace(mi, mx, bins)
    for id, threshold in enumerate(thresholds):
        fr = np.where(genuine <= threshold)[0].shape[0]
        fa = np.where(imposter >= threshold)[0].shape[0]
        frr[id] = fr * 100 / genuine.shape[0]
        far[id] = fa * 100 / imposter.shape[0]

    di = np.argmin(np.abs(far - apcer))
    return round(frr[di], 2)


def get_eer(genuine, imposter, bins=10_001):
    genuine = np.squeeze(np.array(genuine))
    imposter = np.squeeze(np.array(imposter))
    far = np.ones(bins)
    frr = np.ones(bins)
    mi = np.min(imposter)
    mx = np.max(genuine)
    thresholds = np.linspace(mi, mx, bins)
    for id, threshold in enumerate(thresholds):
        fr = np.where(genuine <= threshold)[0].shape[0]
        fa = np.where(imposter >= threshold)[0].shape[0]
        frr[id] = fr * 100 / genuine.shape[0]
        far[id] = fa * 100 / imposter.shape[0]

    di = np.argmin(np.abs(far - frr))

    one = np.argmin(np.abs(far - 1))
    pointone = np.argmin(np.abs(far - 0.1))
    pointzeroone = np.argmin(np.abs(far - 0.01))
    pointzerozeroone = np.argmin(np.abs(far - 0.001))
    eer = (far[di] + frr[di]) / 2
    return (
        round(eer, 2),
        round(100 - frr[one], 2),
        round(100 - frr[pointone], 2),
        round(100 - frr[pointzeroone], 2),
        round(100 - frr[pointzerozeroone], 2),
    )
