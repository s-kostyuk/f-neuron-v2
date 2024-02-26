from typing import Optional, Union

from nns_aaf import LeNetAaf, KerasNetAaf, AfDefinition


# The interval might be suboptimal, but matches the original F-neuron paper.
# For better Sigmoid and Tanh approximation prefer (-4.0; +4.0) with 16 MFs.
AF_FUZZY_DEFAULT_INTERVAL = AfDefinition.AfInterval(
    start=-1.0, end=+1.0, n_segments=12
)

# An interval with improved sensitivity and accuracy.
AF_FUZZY_WIDE_INTERVAL = AfDefinition.AfInterval(
    start=-4.0, end=+4.0, n_segments=16
)


def create_net(
        net_name: str, net_type: str, ds_name: str, af_name: str,
        *, af_name_cnn: Optional[str] = None
) -> Union[LeNetAaf, KerasNetAaf]:

    if af_name_cnn is None:
        af_name_cnn = af_name

    af_name_ffn = af_name

    if net_type.endswith("_shared"):
        af_sharing = True
        net_type = net_type.replace('_shared', '', 1)
    else:
        af_sharing = False

    if net_type == "base":
        af_type_cnn = AfDefinition.AfType.TRAD
        af_type_ffn = AfDefinition.AfType.TRAD
        af_interval_ffn = None
    elif net_type == "ahaf":
        af_type_cnn = AfDefinition.AfType.ADA_AHAF
        af_type_ffn = AfDefinition.AfType.ADA_AHAF
        af_interval_ffn = None
    elif net_type == "ahaf_ffn":
        af_type_cnn = AfDefinition.AfType.TRAD
        af_type_ffn = AfDefinition.AfType.ADA_AHAF
        af_interval_ffn = None
    elif net_type == "leaf":
        af_type_cnn = AfDefinition.AfType.ADA_LEAF
        af_type_ffn = AfDefinition.AfType.ADA_LEAF
        af_interval_ffn = None
    elif net_type == "leaf_ffn":
        af_type_cnn = AfDefinition.AfType.TRAD
        af_type_ffn = AfDefinition.AfType.ADA_LEAF
        af_interval_ffn = None
    elif net_type == "fuzzy_ffn":
        af_type_cnn = AfDefinition.AfType.TRAD
        af_type_ffn = AfDefinition.AfType.ADA_FUZZ
        af_interval_ffn = AF_FUZZY_DEFAULT_INTERVAL
    elif net_type == "fuzzyw_ffn":
        af_type_cnn = AfDefinition.AfType.TRAD
        af_type_ffn = AfDefinition.AfType.ADA_FUZZ
        af_interval_ffn = AF_FUZZY_WIDE_INTERVAL
    else:
        raise ValueError("Network type is not supported")

    cnn_af = AfDefinition(
        af_base=af_name_cnn, af_type=af_type_cnn,
        af_sharing=af_sharing
    )

    ffn_af = AfDefinition(
        af_base=af_name_ffn, af_type=af_type_ffn,
        af_interval=af_interval_ffn,
        af_sharing=af_sharing
    )

    if ds_name == "CIFAR-10":
        ds_name = "CIFAR10"

    if net_name == "KerasNet":
        net = KerasNetAaf(flavor=ds_name, af_conv=cnn_af, af_fc=ffn_af)
    elif net_name == "LeNet-5":
        net = LeNetAaf(flavor=ds_name, af_conv=cnn_af, af_fc=ffn_af)
    else:
        raise NotImplementedError("Only LeNet-5 and KerasNet are supported")

    return net
