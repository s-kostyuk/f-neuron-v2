#!/usr/bin/env python3
import itertools
import argparse
import pprint

from train_common import train_variant
from misc import NetInfo, precision_mng


def main():
    avail_linear_units = ['ReLU', 'SiLU']
    avail_bound_functions = ['Tanh', 'Sigmoid']
    aval_fuzzy_acts = ['Ramp', 'Random', 'Constant']
    avail_networks = ["LeNet-5", "KerasNet"]
    avail_datasets = ["F-MNIST", "CIFAR-10"]
    avail_af_types = ["base",
                      "ahaf", "leaf", "fuzzy_ffn", "fuzzyw_ffn",
                      "ahaf_shared", "leaf_shared", "fuzzy_ffn_shared"]

    parser = argparse.ArgumentParser(
        prog='train_new_simple'
    )
    parser.add_argument('af_type',
                        help='The type of activation functions to use in this '
                             'model (base/fixed, AHAF, LEAF, Fuzzy; in all '
                             'layers or only in the final feed-forward layers; '
                             'individual per neuron or with shared adaptive '
                             'activation weights between neurons)',
                        choices=avail_af_types)
    parser.add_argument('--opt', default='rmsprop',
                        help='The optimizer to use for training',
                        choices=('rmsprop', 'adam'))
    parser.add_argument('--seed', default=42, type=int,
                        help='The initial seed value for random generators')
    parser.add_argument('--bs', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--p24sl', action='store_true',
                        help='Train LEAF with slower learning rates')
    parser.add_argument('--dspu4', action='store_true',
                        help='Train adaptive activations using the Double '
                             'Stage Parameter Update procedure variant 4')
    parser.add_argument('--dev', default='gpu',
                        help='The computing device used for training and eval',
                        choices=('cpu', 'gpu'),)
    parser.add_argument('--net', default='all',
                        help='The neural network models to train',
                        choices=('all', *avail_networks))
    parser.add_argument('--ds', default='all',
                        help='The training dataset',
                        choices=('all', *avail_datasets))
    parser.add_argument('--start_ep', default=0, type=int,
                        help='Load a pre-trained model, pre-trained for the '
                             'specified number of epochs. Init a new model '
                             'from scratch if the start epoch is zero')
    parser.add_argument('--end_ep', default=100, type=int,
                        help='Stop the training and evaluate the model '
                             'performance after this number of epochs')
    parser.add_argument('--patched', action='store_true',
                        help='Load a patched model')
    parser.add_argument('--patch_base', action='store_true',
                        help='Load the base model, patch it with adaptive '
                             'activations, and continue its training')
    parser.add_argument('--patched_from_af', type=str,
                        default=None,
                        help='Indicate the activation function form used in '
                             'the pre-trained model before the patching. By '
                             'default equals to the ones in --acts. Specify '
                             'when patching with incompatible activation '
                             'functions')
    parser.add_argument('--acts', default='all_lus',
                        help='The activation functions to train',
                        choices=('all', 'all_lus', 'all_bfs',
                                 *avail_linear_units, *avail_bound_functions,
                                 'Random', *aval_fuzzy_acts))
    parser.add_argument('--act_cnn', default=None,
                        help='Customize the activation function in CNN layers',
                        choices=(*avail_linear_units, *avail_bound_functions,
                                 'Random'))
    parser.add_argument('--wandb', action='store_true',
                        help='Enable logging to the Weights and Biases service')
    parser.add_argument('--tuned', action='store_true',
                        help='Load a fine-tuned model')
    parser.add_argument('--tune_aaf', action='store_true',
                        help='Fine-tine the adaptive activation functions '
                             'without updating the rest of the model weights')
    parser.add_argument('--precision', type=str, default=None,
                        help='The default floating point precision for all '
                             'PyTorch tensors, used exclusively for logging')
    args = parser.parse_args()

    net_names = avail_networks if args.net == 'all' else [args.net]
    ds_names = avail_datasets if args.ds == 'all' else [args.ds]

    if args.acts == 'all':
        act_names = (*avail_linear_units, *avail_bound_functions)
    elif args.acts == 'all_lus':
        act_names = avail_linear_units
    elif args.acts == 'all_bfs':
        act_names = avail_bound_functions
    else:
        act_names = [args.acts]  # only one specific function

    net_ds_combinations = itertools.product(net_names, ds_names, act_names)
    start_ep = args.start_ep
    end_ep = args.end_ep
    nets = []
    act_cnn = args.act_cnn
    opt = args.opt
    seed = args.seed
    bs = args.bs
    p24sl = args.p24sl
    dspu4 = args.dspu4
    dev_name = args.dev
    patched = args.patched
    patch_base = args.patch_base
    patched_from_af = args.patched_from_af
    wandb = args.wandb
    tuned = args.tuned
    tune_aaf = args.tune_aaf
    precision = args.precision

    for n, ds, act in net_ds_combinations:
        net = NetInfo(n, args.af_type, ds, act, end_ep, dspu4=dspu4,
                      af_name_cnn=act_cnn,
                      opt_name=opt, p24sl=p24sl,
                      patched_from_af=patched_from_af)
        nets.append(net)

    print("Training the following combinations:")
    pprint.pprint(nets)

    for net in nets:
        with precision_mng(precision):
            train_variant(net.net_name, net.net_type, net.ds_name,
                          af_name=net.af_name, af_name_cnn=net.af_name_cnn,
                          end_epoch=net.epoch,
                          start_epoch=start_ep, patched=patched,
                          patched_from_af=net.patched_from_af,
                          dspu4=net.dspu4, p24sl=net.p24sl,
                          opt_name=net.opt_name, seed=seed, bs=bs,
                          dev_name=dev_name, patch_base=patch_base,
                          wandb_enable=wandb, tuned=tuned, tune_aaf=tune_aaf,
                          precision=precision)


if __name__ == "__main__":
    main()
