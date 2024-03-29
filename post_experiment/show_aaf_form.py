import functools
import itertools
import warnings
from typing import Sequence, Tuple, Optional, Callable

import torch
import matplotlib.pyplot as plt
import matplotlib.figure
from cycler import cycler

from adaptive_afs import FNeuronAct
from adaptive_afs import af_build, AfDefinition
from misc import (get_file_name_checkp, get_file_name_aaf_img,
                  get_runs_path,
                  create_net, NetInfo,
                  AF_FUZZY_DEFAULT_INTERVAL, AF_FUZZY_WIDE_INTERVAL)


def get_random_idxs(max_i, cnt=10) -> Sequence[int]:
    return [
        int(torch.randint(size=(1,), low=0, high=max_i)) for _ in range(cnt)
    ]


def random_selection(params, idxs):
    return [params[i] for i in idxs]


def visualize_af_base(af_name: str, x, subfig):
    base_def = AfDefinition(
        af_type=AfDefinition.AfType.TRAD,
        af_base=af_name
    )
    af = af_build(base_def)
    y = af(x)
    x_view = x.cpu().numpy()
    y_view = y.cpu().numpy()
    subfig.plot(x_view, y_view)


def visualize_af_ahaf(rho1, rho3, x, subfig):
    y = (rho1 * x) * torch.sigmoid(rho3 * x)
    x_view = x.cpu().numpy()
    y_view = y.cpu().numpy()
    subfig.plot(x_view, y_view)


def visualize_af_leaf(params, x, subfig):
    rho1, rho2, rho3, rho4 = params
    y = (rho1 * x + rho2) * torch.sigmoid(rho3 * x) + rho4
    x_view = x.cpu().numpy()
    y_view = y.cpu().numpy()
    subfig.plot(x_view, y_view)


def visualize_af_fuzzy(
        fuzzy_def: AfDefinition, x: torch.Tensor, weights: torch.nn.Parameter,
        subfig
):
    def _restore_weights(
            count: int, input_dim: Tuple[int, ...],
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        return weights.data

    af = FNeuronAct(
        left=fuzzy_def.interval.start,
        right=fuzzy_def.interval.end,
        count=weights.size(dim=-1) - 2,
        init_f=_restore_weights
    )
    y = af(x)

    x_view = x.cpu().numpy()
    y_view = y.cpu().numpy()

    subfig.plot(x_view, y_view)


def visualize_afs_ahaf_by_params(
        params: Sequence[torch.nn.Parameter], figbuilder: Callable, layers, show_subtitles,
        reference_af_name: Optional[str] = None,
        rowheight: float = 1.0, colwidth: float = 1.2,
        allow_layers_1row: bool = False
) -> matplotlib.figure.Figure:
    num_neurons = len(params) // 2
    start_index = max(0, num_neurons - layers)
    cols = min(5, len(params[0].view(-1)))
    if cols == 1 and allow_layers_1row:
        transpose = True
    else:
        transpose = False

    x = torch.arange(start=-10, end=4.0, step=0.1,
                     device=params[0].device)

    if transpose:
        width = colwidth * layers
        height = rowheight * cols
    else:
        width = colwidth * cols
        height = rowheight * layers

    fig = figbuilder(figsize=(width, height))

    if transpose:
        subplots = fig.subplots(cols, layers, sharey=True)
    else:
        subplots = fig.subplots(layers, cols, sharey=True)

    # Workaround - 1x1 subplots do not support subscripting
    if cols == 1 and layers == 1:
        subplots = [subplots]

    for i in range(layers):
        param_idx = start_index + i
        all_gamma = params[param_idx * 2].view(-1)
        all_beta = params[param_idx * 2 + 1].view(-1)
        sel = get_random_idxs(max_i=len(all_gamma), cnt=cols)
        sel_gamma = random_selection(all_gamma, sel)
        sel_beta = random_selection(all_beta, sel)

        for j in range(cols):
            if transpose:
                subfig = subplots[j*cols+i]
            else:
                subfig = subplots[i * layers + j]
            if show_subtitles:
                subfig.set_title("L{} F{}".format(i, sel[j]))
                subfig.title.set_size(10)

            gamma = sel_gamma[j]
            beta = sel_beta[j]

            if reference_af_name is not None:
                visualize_af_base(reference_af_name, x, subfig=subfig)

            visualize_af_ahaf(beta, gamma, x, subfig=subfig)

    return fig


def visualize_afs_leaf_by_params(
        params: Sequence[torch.nn.Parameter], figbuilder: Callable, layers, show_subtitles,
        reference_af_name: Optional[str] = None,
        rowheight: float = 1.0, colwidth: float = 1.2,
        allow_layers_1row: bool = False
):
    num_neurons = len(params) // 4
    start_index = max(0, num_neurons - layers)
    cols = min(5, len(params[0].view(-1)))
    if cols == 1 and allow_layers_1row:
        transpose = True
    else:
        transpose = False

    x = torch.arange(start=-10, end=4.0, step=0.1,
                     device=params[0].device)

    if transpose:
        width = colwidth * layers
        height = rowheight * cols
    else:
        width = colwidth * cols
        height = rowheight * layers

    fig = figbuilder(figsize=(width, height))

    if transpose:
        subplots = fig.subplots(cols, layers, sharey=True)
    else:
        subplots = fig.subplots(layers, cols, sharey=True)

    # Workaround - 1x1 subplots do not support subscripting
    if cols == 1 and layers == 1:
        subplots = [subplots]

    for i in range(layers):
        param_idx = start_index + i
        all_p1 = params[param_idx * 4].view(-1)
        all_p2 = params[param_idx * 4 + 1].view(-1)
        all_p3 = params[param_idx * 4 + 2].view(-1)
        all_p4 = params[param_idx * 4 + 3].view(-1)

        sel = get_random_idxs(max_i=len(all_p3), cnt=cols)
        sel_p1 = random_selection(all_p1, sel)
        sel_p2 = random_selection(all_p2, sel)
        sel_p3 = random_selection(all_p3, sel)
        sel_p4 = random_selection(all_p4, sel)

        for j in range(cols):
            if transpose:
                subfig = subplots[j*cols+i]
            else:
                subfig = subplots[i * layers + j]
            if show_subtitles:
                subfig.set_title("L{} F{}".format(i, sel[j]))
                subfig.title.set_size(10)

            instance = sel_p1[j], sel_p2[j], sel_p3[j], sel_p4[j]

            if reference_af_name is not None:
                visualize_af_base(reference_af_name, x, subfig=subfig)

            visualize_af_leaf(instance, x, subfig=subfig)


def visualize_afs_fuzzy_by_params(
        params: Sequence[torch.nn.Parameter], figbuilder: Callable, layers, show_subtitles,
        reference_af_name: Optional[str] = None,
        rowheight: float = 1.0, colwidth: float = 1.2,
        allow_layers_1row: bool = False, fuzzyw: bool = False
) -> matplotlib.figure.Figure:
    num_neurons = len(params) // 1
    start_index = max(0, num_neurons - layers)
    cols = min(5, len(params[0]))
    if cols == 1 and allow_layers_1row:
        transpose = True
    else:
        transpose = False

    # WARNING: Keep the definition updated with create_net
    fuzzy_def = AfDefinition(
        af_base="DoNotCare",  # does not matter for pre-init functions
        af_type=AfDefinition.AfType.ADA_FUZZ,
        af_interval=AF_FUZZY_WIDE_INTERVAL if fuzzyw else AF_FUZZY_DEFAULT_INTERVAL
    )

    # Longer range to visualize Sigmoid and Tanh
    x = torch.arange(start=-3.5, end=+3.5, step=0.1, device=params[0].device)

    if transpose:
        width = colwidth * layers
        height = rowheight * cols
    else:
        width = colwidth * cols
        height = rowheight * layers

    fig = figbuilder(figsize=(width, height))

    if transpose:
        subplots = fig.subplots(cols, layers, sharey=True)
    else:
        subplots = fig.subplots(layers, cols, sharey=True)

    # Workaround - 1x1 subplots do not support subscripting
    if cols == 1 and layers == 1:
        subplots = [subplots]

    for i in range(layers):
        param_idx = start_index + i
        all_mfs_weights = params[param_idx]
        sel = get_random_idxs(max_i=len(all_mfs_weights), cnt=cols)
        sel_mfs_weights = random_selection(all_mfs_weights, sel)

        for j in range(cols):
            if transpose:
                subfig = subplots[j*cols+i]
            else:
                subfig = subplots[i * layers + j]
            if show_subtitles:
                subfig.set_title("L{} F{}".format(i, sel[j]))
                subfig.title.set_size(10)

            weights = sel_mfs_weights[j]

            if reference_af_name is not None:
                visualize_af_base(reference_af_name, x, subfig=subfig)

            visualize_af_fuzzy(fuzzy_def, x, weights, subfig=subfig)

    return fig


def _is_same_af_name(net_info: NetInfo) -> bool:
    if net_info.af_name_cnn is None:
        return True

    if net_info.af_name_cnn == net_info.af_name:
        return True

    if net_info.net_type in ["fuzzy_ffn", "fuzzyw_ffn"]:
        return True

    # leaves AHAF with different AFs in CNN and FFN
    return False


def _get_reference_af_name(net_info: NetInfo) -> Optional[str]:
    if _is_same_af_name(net_info):
        return net_info.af_name
    else:
        warnings.warn("The network contains different activations in the CNN "
                      "and FFN layers. Unable to visualize the base function.")
        return None


def visualize_afs(net_info: NetInfo, max_rows: int = 2, bw=False,
                  show_reference: bool = False):
    torch.manual_seed(seed=128)

    if net_info.net_type.startswith("ahaf"):
        params_per_neuron = 2  # constant for AHAF
        visualizer = visualize_afs_ahaf_by_params
    elif net_info.net_type.startswith("leaf"):
        params_per_neuron = 4  # constant for LEAF
        visualizer = visualize_afs_leaf_by_params
    elif net_info.net_type.startswith("fuzzyw"):
        params_per_neuron = 1  # constant for Fuzzy AF, 1 param set per neuron
        visualizer = functools.partial(
            visualize_afs_fuzzy_by_params, fuzzyw=True
        )
    elif net_info.net_type.startswith("fuzzy"):
        params_per_neuron = 1  # constant for Fuzzy AF, 1 param set per neuron
        visualizer = visualize_afs_fuzzy_by_params
    else:
        raise ValueError("Network type is not supported")

    runs_path = get_runs_path()
    img_path = runs_path + get_file_name_aaf_img(*net_info)
    checkp_path = runs_path + get_file_name_checkp(*net_info)
    checkp = torch.load(checkp_path)
    net = create_net(
        net_info.net_name, net_info.net_type,
        net_info.ds_name, net_info.af_name,
        af_name_cnn=net_info.af_name_cnn
    )
    net.load_state_dict(checkp['net'])

    af_params = net.activation_params
    num_neurons = len(af_params) // params_per_neuron

    show_subtitles = True

    if max_rows is None:
        layers = num_neurons
    else:
        layers = min(max_rows, num_neurons)

    if bw:
        # TODO: Set locally for this figure, not globally
        if show_reference:
            colors = ['black', 'grey']
        else:
            colors = ['grey']
        monochrome = cycler('color', colors)
        plt.rcParams['axes.prop_cycle'] = monochrome

    figbuilder = lambda *args, **kwargs: (
        plt.figure(*args, **kwargs, tight_layout=tight_layout))

    tight_layout = {'pad': 0.35}

    if show_reference:
        ref_af_name = _get_reference_af_name(net_info)
    else:
        ref_af_name = None

    with torch.no_grad():
        fig = visualizer(
            af_params, figbuilder, layers, show_subtitles,
            reference_af_name=ref_af_name,
            rowheight=1.2, colwidth=1.2,
            allow_layers_1row=True
        )

    plt.savefig(img_path, dpi=300, format='svg')
    plt.close(fig)
