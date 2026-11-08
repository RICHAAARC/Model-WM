"""Frozen media and quality operations; no threshold fitting."""
import math
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def mix_rgb(a, b):
    return np.rint(.5*a.astype(np.float32)+.5*b.astype(np.float32)).astype(np.uint8)


def quality(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    # JSON uses null for +infinity and keeps its exact interpretation separately.
    psnr = None if mse == 0 else 10*math.log10(255**2/mse)
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255,
                 gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False,
                 K1=.01, K2=.03))
    return dict(mse=mse, psnr_db=psnr, psnr_infinite=mse == 0, ssim=ssim)


def psnr_value(row):
    return math.inf if row["psnr_infinite"] else row["psnr_db"]


def native_gate(rows, gates):
    # Exactly sixteen members from each source; missing rows never shrink denominator.
    result = {}
    for source in ["registered", "foreign"]:
        group = [x for x in rows if x["source"] == source]
        complete = len(group) == 16 and len({x["trajectory_id"] for x in group}) == 16
        accuracy = sum(x["correct_bits"] for x in group)/(16*48)
        mean_psnr = sum(psnr_value(x) for x in group)/16 if complete else None
        mean_ssim = sum(x["ssim"] for x in group)/16 if complete else None
        count20 = sum(psnr_value(x) >= gates["each_native_psnr_db"] for x in group)
        passed = bool(complete and accuracy >= gates["native_bit_accuracy"] and
                      mean_psnr >= gates["mean_psnr_db"] and mean_ssim >= gates["mean_ssim"] and count20 == 16)
        result[source] = dict(denominator=16, observed=len(group), bit_denominator=768,
                              bit_accuracy=accuracy, mean_psnr_db=None if mean_psnr == math.inf else mean_psnr,
                              mean_psnr_infinite=mean_psnr == math.inf, mean_ssim=mean_ssim,
                              native_psnr_at_least20=count20, passed=passed)
    return result


def interaction(images, logits):
    k, j, h, jh = [images[n].astype(np.float64)/255 for n in ["probe_k", "probe_j", "probe_h", "probe_jh"]]
    rms = lambda a: float(np.sqrt(np.mean(a*a)))
    absolute = rms(jh-j-h+k)
    first = .5*(rms(j-k)+rms(h-k))
    # Diagnostic only; no success threshold for this single DEV probe.
    return dict(second_order_rms=absolute, mean_first_order_rms=first,
                relative_interaction=None if first < 1/255 else absolute/first,
                status="UNRESOLVED" if first < 1/255 else "DESCRIPTIVE_ONLY",
                logits_second_order_rms=rms(logits["probe_jh"]-logits["probe_j"]-logits["probe_h"]+logits["probe_k"]))
