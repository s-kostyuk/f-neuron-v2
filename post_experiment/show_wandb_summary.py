import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl

base_path = "runs"

file_names_tgt = [
    'lenet5_f-mnist',
    'lenet5_cifar10',
    'kerasnet_f-mnist',
    'kerasnet_cifar10',
]

net_names = [
    'LeNet-5',
    'LeNet-5',
    'KerasNet',
    'KerasNet',
]

file_names_src_lenet5_fmnist = [
    f'{base_path}/wandb_lenet5_fmnist_relu.csv',
    f'{base_path}/wandb_lenet5_fmnist_ramp.csv',
    f'{base_path}/wandb_lenet5_fmnist_rand.csv',
    f'{base_path}/wandb_lenet5_fmnist_const.csv',
]

file_names_src_lenet5_cifar10 = [
    f'{base_path}/wandb_lenet5_cifar10_relu.csv',
    f'{base_path}/wandb_lenet5_cifar10_ramp.csv',
    f'{base_path}/wandb_lenet5_cifar10_rand.csv',
    f'{base_path}/wandb_lenet5_cifar10_const.csv',
]

file_names_src_kerasnet_fmnist = [
    f'{base_path}/wandb_kerasnet_fmnist_relu.csv',
    f'{base_path}/wandb_kerasnet_fmnist_ramp.csv',
    f'{base_path}/wandb_kerasnet_fmnist_rand.csv',
    f'{base_path}/wandb_kerasnet_fmnist_const.csv',
]

file_names_src_kerasnet_cifar10 = [
    f'{base_path}/wandb_kerasnet_cifar10_relu.csv',
    f'{base_path}/wandb_kerasnet_cifar10_ramp.csv',
    f'{base_path}/wandb_kerasnet_cifar10_rand.csv',
    f'{base_path}/wandb_kerasnet_cifar10_const.csv',
]

file_names_src = [
    file_names_src_lenet5_fmnist,
    file_names_src_lenet5_cifar10,
    file_names_src_kerasnet_fmnist,
    file_names_src_kerasnet_cifar10,
]

legends = [
    'ReLU',
    'Fuzzy Ramp-like',
    'Fuzzy Random-like',
    'Fuzzy Constant-like',
]

for srcs, tgt, net_name in zip(file_names_src, file_names_tgt, net_names):
    ax = None

    for src in srcs:
        df = pd.read_csv(src)
        col_name = f"net_name: {net_name} - test_acc"
        col_name_min = f"{col_name}__MIN"
        col_name_max = f"{col_name}__MAX"

        ax = df.plot(ax=ax, x='Step', y=col_name, xlabel="Epoch",
                     ylabel="Accuracy, %", figsize=(7, 3.5))
        color = plt.gca().lines[-1].get_color()
        ax.fill_between(x='Step', y1=col_name_min, y2=col_name_max, data=df,
                        color=mpl.colors.to_rgba(color, 0.15))

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.legend(legends)
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=0.98,
                        wspace=0.25)

    del ax
    plt.savefig(f"{base_path}/wandb_summary_{tgt}.svg")
