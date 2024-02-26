import json
import warnings

from typing import Optional, Callable, List, Union, Iterable, Dict, Tuple

import torch
import torch.nn
import torch.utils.data
import torchinfo

import random
import numpy

from experiments.tune_patched_common import get_params_freeze_status

try:
    import wandb
    WANDB_AVAILABLE = True
    import platform
except ImportError:
    WANDB_AVAILABLE = False

from adaptive_afs import LEAF
from experiments.common import get_device, get_dataset
from misc import get_file_name_checkp, get_file_name_stat,\
    get_file_name_train_args, get_runs_path, CheckPoint

from nns_aaf import KerasNetAaf, LeNetAaf
from misc import RunningStat, ProgressRecorder, create_net


AafNetwork = Union[KerasNetAaf, LeNetAaf]


def _net_train_aaf(net: AafNetwork):
    for p in net.parameters():
        p.requires_grad = False

    for p in net.activation_params:
        p.requires_grad = True


def _net_train_non_aaf(net: AafNetwork):
    for p in net.parameters():
        p.requires_grad = True

    for p in net.activation_params:
        p.requires_grad = False


def _net_train_noop(net: AafNetwork):
    pass


def _net_split_leaf_params(net: AafNetwork):
    aaf_param_ids = set()
    leaf_p24_params = []
    aaf_rest_params = []
    generic_params = []

    for act in net.activations:
        if isinstance(act, LEAF):
            p24_params = (act.p2, act.p4)
            p13_params = (act.p1, act.p3)

            leaf_p24_params.extend(p24_params)
            aaf_rest_params.extend(p13_params)
            aaf_param_ids.update(
                (id(p) for p in act.parameters())
            )
        elif isinstance(act, torch.nn.Module):
            aaf_rest_params.extend(act.parameters())
            aaf_param_ids.update(
                (id(p) for p in act.parameters())
            )

    for p in net.parameters():
        if id(p) not in aaf_param_ids:
            generic_params.append(p)

    return leaf_p24_params, aaf_rest_params, generic_params


def get_opt_by_name(
        opt_name: str, base_lr: float,
        net_params: Iterable[Union[torch.nn.Parameter, Dict]]
) -> torch.optim.Optimizer:
    if opt_name == 'rmsprop':
        opt = torch.optim.RMSprop(
            params=net_params,
            lr=base_lr,
            alpha=0.9,  # default Keras
            momentum=0.0,  # default Keras
            eps=1e-7,  # default Keras
            centered=False  # default Keras
        )
    elif opt_name == 'adam':
        opt = torch.optim.Adam(
            params=net_params,
            lr=base_lr,
        )
    else:
        raise NotImplementedError("Only ADAM and RMSProp supported")

    return opt


def _eval_net(net, dev, test_loader, error_fn) -> Tuple[float, float]:
    test_total = 0
    test_correct = 0
    test_loss = -1.0

    net.eval()

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(dev)
            y = batch[1].to(dev)
            y_hat = net(x)
            test_loss = error_fn(y_hat, target=y)
            _, pred = torch.max(y_hat.data, 1)
            test_total += y.size(0)
            test_correct += torch.eq(pred, y).sum().item()

    test_acc = test_correct / test_total
    return test_acc, test_loss


def train_variant(
        net_name: str, net_type: str,
        ds_name: str, af_name: str, end_epoch: int = 100, *,
        start_epoch: int = 0,
        patched: bool = False,
        patched_from_af: Optional[str] = None,
        patched_from_af_cnn: Optional[str] = None,
        af_name_cnn: Optional[str] = None,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop',
        seed: int = 42, bs: int = 64, dev_name: str = 'gpu',
        patch_base: bool = False, wandb_enable: bool = False,
        tuned: bool = False, tune_aaf: bool = False,
        precision: str = None
):
    """
    Initialize, load and train the model for the specified number of epochs.
    Saves the trained network, the optimizer state and the statistics to the
    `./runs` directory.

    :param net_name: name of the model - "KerasNet" or "LeNet-5"
    :param net_type: type of the model - "base", "ahaf", "leaf", "fuzzy_ffn",
                     "fuzzyw_ffn"
    :param ds_name: name of the dataset - "CIFAR-10" or "F-MNIST"
    :param af_name: the initial activation function form name -
                    "ReLU", "SiLU", "Tanh", "Sigmnoid" and so on
    :param end_epoch: stop the training at this epoch
    :param start_epoch: start the training at this epoch
    :param patched: indicates to load the "patched" model that was initially
                    trained with the base activation and then upgraded to an
                    adaptive alternative
    :param patched_from_af: the activation function form used in the pre-trained
           model before the patching, by default, equals to the one in af_name
    :param patched_from_af_cnn: the activation function form used in the
           pre-trained model before the patching for CNN layers, by default,
           equals to the one in af_name_cnn, if af_name_cnn is undefined - to
           af_name
    :param af_name_cnn: specify the different initial activation function form
                        for the fully connected layers of the network
    :param dspu4: set to `True` to use the 2SPU-4 training procedure
    :param p24sl: set to `True` to decrease LR for LEAF params p2 and p4
    :param opt_name: set the optimizer: 'adam' or 'rmsprop'
    :param seed: the initial value for RNG
    :param bs: the training data block size
    :param dev_name: training executor device
    :param patch_base: perform in-place patching of the base network
    :param wandb_enable: enable logging to Weights and Biases
    :param tuned: indicates to load the "tuned" model that has fine-tuned AAFs
                  after the activation function replacement
    :param tune_aaf: set to `True` to only fine-tune AAFs without updating
                     non-AAF parameters
    :param precision: the current default data type and precision, only for logs
    :return: None
    """
    if wandb_enable and not WANDB_AVAILABLE:
        raise ValueError(
            "The wandb library is not available. Install the library or disable"
            "logging to Weights and Biases in the arguments"
        )

    batch_size = bs
    rand_seed = seed
    runs_path = get_runs_path()

    dev = get_device(dev_name)
    torch.manual_seed(rand_seed)
    torch.use_deterministic_algorithms(mode=True)

    train_set, test_set, dims = get_dataset(ds_name, augment=True)
    input_size = (batch_size, *dims)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, num_workers=4
    )

    net = create_net(
        net_name, net_type, ds_name, af_name, af_name_cnn=af_name_cnn
    )

    error_fn = torch.nn.CrossEntropyLoss()
    net.to(device=dev)
    torchinfo.summary(net, input_size=input_size, device=dev)

    if opt_name == 'rmsprop':
        base_lr = 1e-4
        p24lr = base_lr / 10
    elif opt_name == 'adam':
        base_lr = 0.001
        p24lr = base_lr / 1000
    else:
        raise NotImplementedError("Only ADAM and RMSProp supported")

    if not net_type.startswith("leaf"):
        # Ignore on everything except LEAF
        p24sl = False

    net_params_leaf_p24, net_params_aaf_rest, net_params_non_aaf = _net_split_leaf_params(net)
    opt_params_non_aaf = [
        {'params': net_params_non_aaf}
    ]

    if p24sl:
        print(f"Using a custom learning rate of {p24lr} for LEAF params "
              f"p2 and p4")
        opt_params_aaf = [
            {'params': net_params_aaf_rest},
            {'params': net_params_leaf_p24, 'lr': p24lr}
        ]
    else:
        opt_params_aaf = [
            {'params': [*net_params_aaf_rest, *net_params_leaf_p24]}
        ]

    opt_params_sets: List[List[Dict]]

    if dspu4:
        # Create two different optimizers: one for AAF, one for non-AAF params
        opt_params_sets = [
            opt_params_aaf, net_params_non_aaf
        ]
    else:
        # Create a single optimizer for AAF and non-AAF params
        opt_params_sets = [
            [*opt_params_non_aaf, *opt_params_aaf]
        ]

    opts: List[torch.optim.Optimizer]
    opts = [get_opt_by_name(opt_name, base_lr, ps) for ps in opt_params_sets]

    if start_epoch > 0:
        net_type_to_load = "base" if patch_base else net_type
        strict_load = not patch_base
        dspu4_to_load = False if patch_base else dspu4
        p24sl_to_load = False if patch_base else p24sl

        if patch_base and patched_from_af:
            af_name_to_load = patched_from_af
            af_name_cnn_to_load = (patched_from_af_cnn
                                   if patched_from_af_cnn
                                   else patched_from_af)
        else:
            af_name_to_load = af_name
            af_name_cnn_to_load = af_name_cnn

        checkp_name = get_file_name_checkp(
            net_name, net_type_to_load, ds_name, af_name_to_load,
            start_epoch, patched,
            af_name_cnn=af_name_cnn_to_load,
            patched_from_af=patched_from_af,
            patched_from_af_cnn=patched_from_af_cnn,
            dspu4=dspu4_to_load,
            p24sl=p24sl_to_load, opt_name=opt_name, fine_tuned=tuned
        )
        checkp_path = runs_path + checkp_name

        checkp: CheckPoint
        checkp = torch.load(checkp_path)
        net.load_state_dict(checkp['net'], strict=strict_load)

        if (checkp.get('opts')) and (not patch_base):
            opt_states = checkp['opts']
            assert len(opts) == len(opt_states)
            for i in range(len(opt_states)):
                opts[i].load_state_dict(opt_states[i])
        else:
            warnings.warn(
                "The old optimizer state is not available{}. Initialized the "
                "optimizer from scratch.".format(
                    " after patching" if (patched or patch_base) else ""
                )
            )

    print(
        "Training the {} {} network with {} in CNN and {} in FFN "
        "on the {} dataset for {} epochs total using the {} training procedure "
        "and the {} optimizer."
        "".format(
            net_type, net_name, af_name if af_name_cnn is None else af_name_cnn,
            af_name, ds_name, end_epoch, "2SPU-4" if dspu4 else "standard",
            opt_name
        )
    )

    mb_param_freezers: List[Union[None, Callable[[AafNetwork], None]]]
    mb_param_freezers = []

    if dspu4 and tune_aaf:
        # Train AAF params only, skip training for non-AAF params
        mb_param_freezers.append(_net_train_aaf)
        mb_param_freezers.append(None)
    elif dspu4 and not tune_aaf:
        # Train AAF params, then non-AAF params
        mb_param_freezers.append(_net_train_aaf)
        mb_param_freezers.append(_net_train_non_aaf)
    elif not dspu4 and tune_aaf:
        # Freeze non-AAF params
        mb_param_freezers.append(_net_train_aaf)
    else:
        # No freezing
        mb_param_freezers.append(_net_train_noop)

    assert len(opts) == len(mb_param_freezers)

    progress = ProgressRecorder()

    # TODO: Refactor, pass TypedDict as the function argument
    args_content = {
        "net_name": net_name,
        "net_type": net_type,
        "ds_name": ds_name,
        "af_name": af_name,
        "end_epoch": end_epoch,
        "start_epoch": start_epoch,
        "patched": patched,
        "af_name_cnn": af_name_cnn,
        "dspu4": dspu4,
        "p24sl": p24sl,
        "opt_name": opt_name,
        "seed": seed,
        "bs": bs,
        "dev_name": dev_name,
        "patch_base": patch_base,
        "patched_from_af": patched_from_af,
        "patched_from_af_cnn": patched_from_af_cnn,
        "tuned": tuned,
        "tune_aaf": tune_aaf,
        "precision": precision
    }

    args_name = get_file_name_train_args(
        net_name, net_type, ds_name, af_name, end_epoch,
        patched or patch_base,
        fine_tuned=tuned or tune_aaf, af_name_cnn=af_name_cnn,
        dspu4=dspu4, p24sl=p24sl, opt_name=opt_name,
        patched_from_af=patched_from_af, patched_from_af_cnn=patched_from_af_cnn
    )
    args_path = runs_path + args_name

    if wandb_enable:
        wandb_exec_node = platform.node()
        wandb_run_name = args_name
        wandb_run_name = wandb_run_name.rstrip("_args.json")
        wandb_run_name = f"{wandb_run_name}_seed{seed}_{wandb_exec_node}"
        wandb_group_name = f"{net_name}_{ds_name}_{opt_name}_bs{bs}"
        wandb.init(
            project='f-neuron-v2', reinit=True, name=wandb_run_name,
            config=args_content, group=wandb_group_name
        )
        wandb.watch(net, criterion=error_fn, log_freq=390, log='all')

    with open(args_path, 'w') as f:
        json.dump(args_content, f, indent=2)

    # Allows printing debug information only on the first minibatch
    first_mb = True

    # Re-seed after initializing the network. Ensures the same input data stream
    # across the runs.
    torch.manual_seed(rand_seed)

    for epoch in range(start_epoch, end_epoch):
        net.train()
        loss_stat = RunningStat()
        progress.start_ep()

        for mb in train_loader:
            x, y = mb[0].to(dev), mb[1].to(dev)
            last_loss_in_mb: float = -1.0

            for mbf, opt in zip(mb_param_freezers, opts):
                if mbf is None:
                    continue  # skip iteration with this optimizer
                else:
                    mbf(net)

                if first_mb:
                    # Debug information - the list of non-frozen params
                    print(
                        "Active params:",
                        get_params_freeze_status(net.parameters())
                    )

                # The wandb logger does not support `net.forward()`
                y_hat = net(x)
                loss = error_fn(y_hat, target=y)
                last_loss_in_mb = loss.item()

                # Update parameters
                opt.zero_grad()
                loss.backward()
                opt.step()

            first_mb = False
            loss_stat.push(last_loss_in_mb)

        progress.end_ep()

        test_acc, test_loss = _eval_net(net, dev, test_loader, error_fn)
        print("Train set loss stat: m={}, var={}".format(
            loss_stat.mean, loss_stat.variance
        ))
        print("Epoch: {}. Test set accuracy: {:.2%}. Test set loss: {}".format(
                epoch, test_acc, test_loss
        ))
        if wandb_enable:
            wandb.log({
                'train_ep': epoch,
                'train_loss': loss_stat.mean,
                'test_loss': test_loss,
                'test_acc': test_acc}
            )
        progress.push_ep(
            epoch, loss_stat.mean, loss_stat.variance, test_acc,
            lr=' '.join(
                [str(pg["lr"]) for opt in opts for pg in opt.param_groups]
            )
        )

    if start_epoch == end_epoch:
        test_acc, test_loss = _eval_net(net, dev, test_loader, error_fn)
        print(
            "Evaluation-only mode. Epoch: {}. Test set accuracy: {:.2%}. "
            "Test set loss: {}".format(
                start_epoch, test_acc, test_loss
            ))
        progress.push_ep(start_epoch, -1.0, 0.0, test_acc, lr='')

    progress_name = get_file_name_stat(
        net_name, net_type, ds_name, af_name, end_epoch,
        patched or patch_base,
        fine_tuned=tuned or tune_aaf, af_name_cnn=af_name_cnn,
        dspu4=dspu4, p24sl=p24sl, opt_name=opt_name,
        patched_from_af=patched_from_af, patched_from_af_cnn=patched_from_af_cnn
    )
    progress_path = runs_path + progress_name

    if start_epoch != end_epoch or patch_base:
        # Do not override the progress in the eval-only mode
        progress.save_as_csv(progress_path)

    checkp: CheckPoint
    checkp = {
        'net': net.state_dict(),
        'opts': [opt.state_dict() for opt in opts]
    }
    checkp_name = get_file_name_checkp(
        net_name, net_type, ds_name, af_name, end_epoch,
        patched or patch_base,
        fine_tuned=tuned or tune_aaf, af_name_cnn=af_name_cnn,
        dspu4=dspu4, p24sl=p24sl, opt_name=opt_name,
        patched_from_af=patched_from_af, patched_from_af_cnn=patched_from_af_cnn
    )
    checkp_path = get_runs_path() + checkp_name

    torch.save(
        checkp,
        checkp_path
    )

    if wandb_enable:
        wandb.save(checkp_path)
        wandb.finish()
