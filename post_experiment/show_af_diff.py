import itertools
import time
from typing import Optional, NamedTuple

import torch
import torch.nn.functional
import torch.backends.cuda
import torch.backends.cudnn
from matplotlib import pyplot as plt
from cycler import cycler

from adaptive_afs import af_build, AfDefinition
from adaptive_afs.trad import tanh_manual, silu_manual
from misc import precision_mng, get_runs_path


class ErrorHolder(NamedTuple):
    min_error: float
    max_error: float
    mean_error: float


def estimate_error(
        orig_fn, drv_fn, left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    n_points = 100000

    range_len = right - left
    step = range_len / n_points
    eps = step / 100

    with torch.no_grad():
        x = torch.arange(start=left, end=right + eps, step=step)

        y = orig_fn(x)
        y_hat = drv_fn(x)

        y_nans = torch.isnan(y)
        y_hat_nans = torch.isnan(y_hat)

        if y_nans.any() or y_hat_nans.any():
            print(torch.masked_select(x, y_nans), torch.masked_select(x, y_hat_nans))
            raise ValueError(torch.masked_select(x, y_nans), torch.masked_select(x, y_hat_nans))

        errors = torch.square(y - y_hat)
        max_error = torch.max(errors)
        min_error = torch.min(errors)
        mean_error = torch.mean(errors)

        x_view = x.cpu().numpy()
        err_view = errors.cpu().numpy()

        monochrome = cycler('color', ['black'])
        plt.rcParams['axes.prop_cycle'] = monochrome

        tight_layout = {'pad': 0.35}
        plt.figure(tight_layout=tight_layout, figsize=(2, 4))

        plt.xlabel("Input, x")
        plt.ylabel("Error, E=Δ^2")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title(
            f"min={min_error.item():.2e},\nmax={max_error.item():.2e}",
            fontsize=10,
            pad=20,
            loc='left'
        )

        plt.plot(x_view, err_view)

        if img_path is None:
            plt.show()
        else:
            plt.savefig(img_path, dpi=300, format='svg')
            plt.close()

    return ErrorHolder(min_error.item(), max_error.item(), mean_error.item())


def estimate_err_manual_silu(
        left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    orig_fn = torch.nn.functional.silu
    drv_fn = silu_manual

    return estimate_error(orig_fn, drv_fn, left, right, img_path)


def estimate_err_manual_tanh(
        left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    orig_fn = torch.tanh
    drv_fn = tanh_manual

    return estimate_error(orig_fn, drv_fn, left, right, img_path)


def estimate_err_aaf(
        af_def: AfDefinition,
        left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    orig_fn = af_build(
        AfDefinition(af_def.af_base, AfDefinition.AfType.TRAD)
    )
    drv_fn = af_build(af_def)

    return estimate_error(orig_fn, drv_fn, left, right, img_path)


def estimate_all(dev_name: str, prec_name: str):
    runs_path = get_runs_path()
    estimate_err_manual_silu(
        -15.0, +15.0,
        img_path=f"{runs_path}af_diff_manual_silu_{dev_name}_{prec_name}.svg"
    )

    #estimate_err_manual_tanh(
    #    -15.0, +15.0,
    #    img_path=f"{runs_path}af_diff_manual_tanh_{dev_name}_{prec_name}.svg"
    #)

    af_defs_fuzz = [
        AfDefinition(af_base="Tanh", af_type=AfDefinition.AfType.ADA_FUZZ,
                     af_interval=AfDefinition.AfInterval(
                         start=-4.0, end=+4.0, n_segments=16
                     ))
    ]

    for ff in af_defs_fuzz:
        img_path = f"{runs_path}af_diff_fuzz_{ff.af_base}_{dev_name}_{prec_name}.svg"
        estimate_err_aaf(ff, ff.interval.start - 1, ff.interval.end + 1, img_path)

    af_defs_fuzz_precise = [
        AfDefinition(af_base="Tanh", af_type=AfDefinition.AfType.ADA_FUZZ,
                     af_interval=AfDefinition.AfInterval(
                         start=-4.0, end=+4.0, n_segments=128
                     ))
    ]

    for ff in af_defs_fuzz_precise:
        img_path = f"{runs_path}af_diff_fuzz_precise_{ff.af_base}_{dev_name}_{prec_name}.svg"
        estimate_err_aaf(ff, ff.interval.start - 1, ff.interval.end + 1, img_path)

    af_names_ahaf = ["ReLU", "SiLU"]

    for afn in af_names_ahaf:
        img_path = f"{runs_path}af_diff_ahaf_{afn}_{dev_name}_{prec_name}.svg"
        af_def = AfDefinition(af_base=afn, af_type=AfDefinition.AfType.ADA_AHAF)
        estimate_err_aaf(af_def, -15.0, +15.0, img_path)

    af_names_leaf = ["ReLU", "SiLU", "Tanh", "Sigmoid"]

    for afn in af_names_leaf:
        img_path = f"{runs_path}af_diff_leaf_{afn}_{dev_name}_{prec_name}.svg"
        af_def = AfDefinition(af_base=afn, af_type=AfDefinition.AfType.ADA_LEAF)
        estimate_err_aaf(af_def, -15.0, +15.0, img_path)


def main():
    devices = ["cpu", "cuda"]
    precisions = ["float16", "float16rpr", "float32", "tf32", "float64"]

    for dev, prec in itertools.product(devices, precisions):
        if dev == "cpu" and prec in ["float16", "float16rpr", "tf32"]:
            # Skip, not implemented in PyTorch
            continue

        with torch.device(dev):
            with precision_mng(prec):
                eval_start = time.time()
                estimate_all(dev, prec)
                eval_end = time.time()
                eval_time = eval_end - eval_start
                print(dev, prec, eval_time)


if __name__ == "__main__":
    main()
