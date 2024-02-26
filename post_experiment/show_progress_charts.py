#!/usr/bin/env python3

import itertools
import csv
import matplotlib.pyplot as plt

from typing import Sequence, Tuple, List, Union
from cycler import cycler

from misc import ProgressElement
from misc import NetInfo
from misc import get_file_name_stat, get_file_name_stat_img, get_runs_path


def load_results(file_path: str) -> List[ProgressElement]:
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_short_af_name(orig: str) -> str:
    if orig == "Tanh":
        return "tanh"
    elif orig == "Sigmoid":
        return "σ-fn"
    else:
        return orig


def get_legend_short(
        net_info: Union[Tuple, NetInfo], omit_af_names: bool = False,
        include_opt: bool = False
) -> str:
    if not isinstance(net_info, NetInfo):
        net_info = NetInfo(*net_info)

    af_name_ffn = net_info.af_name

    if net_info.af_name_cnn is None:
        af_name_cnn = af_name_ffn
    else:
        af_name_cnn = net_info.af_name_cnn

    af_name_cnn = get_short_af_name(af_name_cnn)
    af_name_ffn = get_short_af_name(af_name_ffn)

    if net_info.net_type.startswith("base"):
        net_type_str = "Base"
    elif net_info.net_type.startswith("ahaf"):
        net_type_str = "AHAF"
    elif net_info.net_type.startswith("leaf"):
        net_type_str = "LEAF"
    elif net_info.net_type.startswith("fuzzy"):
        net_type_str = "Fuzzy"
    else:
        raise ValueError("Network type is not supported")

    if "_ffn" in net_info.net_type:
        net_type_str += " FFN"

    if "_shared" in net_info.net_type:
        net_type_str += " Shared"

    if omit_af_names:
        legend = net_type_str
    else:
        legend = f"{net_type_str}, {af_name_cnn}, {af_name_ffn}"

    if net_info.opt_name == 'adam':
        opt_name_str = 'ADAM'
    elif net_info.opt_name == 'rmsprop':
        opt_name_str = 'RMSprop'
    else:
        raise ValueError("Optimizer is not supported")

    if include_opt:
        legend = f"{legend}, {opt_name_str}"

    if net_info.fine_tuned:
        legend = legend + ", tuned"

    if net_info.dspu4:
        legend = legend + ", DSPT"

    if net_info.p24sl:
        legend = legend + ", P24Sl"

    return legend


def analyze_network(
        net_info: Tuple, omit_af_names: bool = False, include_opt: bool = False
):
    runs_path = get_runs_path()
    file_path = runs_path + get_file_name_stat(*net_info)
    results = load_results(file_path)
    base_legend = get_legend_short(net_info, omit_af_names, include_opt)

    acc = []
    loss = []

    for r in results:
        acc.append(float(r["test_acc"]) * 100.0)
        loss.append(float(r["train_loss_mean"]))

    return base_legend, acc, loss


def plot_networks(
        fig, nets: Sequence[Union[Tuple, NetInfo]],
        bw=False, omit_af_names=False, include_opt=False
) -> bool:
    legends = []

    monochrome = (
            cycler('linestyle', ['-', '--', ':', '-.'])
            * cycler('color', ['black', 'grey'])
            * cycler('marker', ['None'])
    )

    acc_fig, loss_fig = fig.subplots(1, 2)
    #acc_loc = plticker.LinearLocator(numticks=10)
    #acc_fig.yaxis.set_major_locator(acc_loc)
    acc_fig.set_xlabel('epoch')
    acc_fig.set_ylabel('test accuracy, %')
    acc_fig.grid()
    if bw:
        acc_fig.set_prop_cycle(monochrome)

    #loss_loc = plticker.LinearLocator(numticks=10)
    #loss_fig.yaxis.set_major_locator(loss_loc)
    loss_fig.set_xlabel('epoch')
    loss_fig.set_ylabel('training loss')
    loss_fig.grid()
    if bw:
        loss_fig.set_prop_cycle(monochrome)

    net_processed = 0

    for net in nets:
        try:
            base_legend, acc, loss = analyze_network(
                net, omit_af_names, include_opt
            )
        #except FileNotFoundError:
        #    continue
        except Exception as e:
            print("Exception: {}, skipped".format(e))
            continue

        net_processed += 1
        n_epochs = len(acc)
        end_ep = net.epoch
        start_ep = end_ep - n_epochs

        x = tuple(range(start_ep, end_ep))

        legends.append(
            base_legend
        )
        acc_fig.plot(x, acc)
        loss_fig.plot(x, loss)

    fig.legend(legends, bbox_to_anchor=(0.5, 0), loc='lower center', ncol=2)
    fig.subplots_adjust(top=0.95, bottom=0.35, left=0.075, right=0.98, wspace=0.25)

    return net_processed > 0


def alternate_els(list1: list, list2: list) -> list:
    return [ el for pair in zip(list1, list2) for el in pair ]


def visualize(
        net_name: str, ds_name: str, net_group: str,
        nets: Sequence[Union[Tuple, NetInfo]], base_title=None,
        bw: bool = False, omit_af_names: bool = False,
        include_opt: bool = False
):
    fig = plt.figure(figsize=(7, 3.5))
    if base_title is not None:
        title = "{}, test accuracy and training loss".format(base_title)
        fig.suptitle(title)

    success = plot_networks(fig, nets, bw, omit_af_names, include_opt)
    if success:
        #plt.show()
        runs_path = get_runs_path()
        fig_path = runs_path + get_file_name_stat_img(
            net_name, ds_name, net_group,
            nets[0].epoch, nets[0].patched,
            nets[0].fine_tuned
        )
        plt.savefig(fig_path)

    plt.close(fig)


def main():
    net_names = ["LeNet-5", "KerasNet"]
    ds_names = ["F-MNIST", "CIFAR-10"]
    net_ds_combinations = itertools.product(net_names, ds_names)
    opt = "rmsprop"
    black_white = True

    omit_all_captions = True

    for n, ds in net_ds_combinations:
        curr_group = [
            NetInfo(n, "base", ds, "ReLU", 100, opt_name=opt),
        ]

        visualize(
            n, ds, f"{n}_{ds}", curr_group,
            None if omit_all_captions else "{n} on {ds}",
            bw=black_white, omit_af_names=False
        )


if __name__ == "__main__":
    main()
