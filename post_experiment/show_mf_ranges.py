import torch

import matplotlib.pyplot as plt
from cycler import cycler

from adaptive_afs import af_build, AfDefinition
from post_experiment.show_aaf_form import visualize_af_base, visualize_af_fuzzy


def show_f_neuron_mfs(f_neuron):
    mfs = f_neuron._mfs
    mfws = f_neuron._weights[0]

    x_left = -6.5
    x_right = +6.5
    x_len = x_right - x_left
    x_datapoints = 4096
    x_interval = x_len / x_datapoints

    xs = torch.arange(x_left, x_right, x_interval)
    fig = plt.figure(figsize=(5, 2.5))
    ax = fig.subplots(1, 1)

    monochrome = (
            cycler('color', ['black', 'grey'])
    )
    ax.set_prop_cycle(monochrome)
    ax.set_xlabel('Internal activation, u')
    ax.set_ylabel('Synapse output, y')

    with torch.no_grad():
        for mf, mfw in zip(mfs, mfws):
            ys = mf(xs) * mfw
            mask = ys > 0.0000001
            x_view = xs.cpu().numpy()[mask]
            y_view = ys.cpu().numpy()[mask]
            ax.plot(x_view, y_view)

    fig.subplots_adjust(top=0.95, bottom=0.20, left=0.12, right=0.98,
                        wspace=0.25)

    plt.show()


def show_f_neuron(f_neuron):
    x_left = -6.5
    x_right = +6.5
    x_len = x_right - x_left
    x_datapoints = 4096
    x_interval = x_len / x_datapoints

    x = torch.arange(x_left, x_right, x_interval)
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.subplots(1, 1)

    monochrome = (
            cycler('color', ['black', 'grey'])
    )
    ax.set_prop_cycle(monochrome)
    ax.set_xlabel('Internal activation, u')
    ax.set_ylabel('Synapse output, y')

    visualize_af_base("Sigmoid", x, subfig=ax)

    with torch.no_grad():
        y = f_neuron(x)
        x_view = x.cpu().numpy()
        y_view = y.cpu().numpy()
        ax.plot(x_view, y_view)

    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=0.98,
                        wspace=0.25)

    plt.show()


def main():
    f_neuron_def = AfDefinition(
        af_base="Constant", af_type=AfDefinition.AfType.ADA_FUZZ,
        af_interval=AfDefinition.AfInterval(-6.0, +6.0, 12)
    )
    f_neuron = af_build(f_neuron_def)

    mfs = f_neuron._mfs
    mfws = f_neuron._weights[0]

    x_left = -6.5
    x_right = +6.5
    x_len = x_right - x_left

    show_f_neuron_mfs(f_neuron)

    loss_fn = torch.nn.MSELoss()
    base_lr = 1.0
    compensation = 0.5  # the loss criteria in paper is 0.5*MSE(y,yhat)
    lr = base_lr * compensation

    opt = torch.optim.SGD(params=f_neuron.parameters(), lr=lr)

    ############################################################################
    # Manual training example                                                  #
    ############################################################################

    x1 = torch.Tensor([-6.5])
    yhat1 = f_neuron(x1)
    y1 = torch.sigmoid(x1)
    loss1 = loss_fn(yhat1, target=y1)

    opt.zero_grad()
    loss1.backward()
    opt.step()
    del loss1

    print("Params, step 1:", [*f_neuron.parameters()])

    x2 = torch.Tensor([-5.5])

    with torch.no_grad():
        y = [mf.forward(x2) for mf in mfs]
        print("MFs, step 2:", y)

        y = [w * mf.forward(x2) for w, mf in zip(mfws, mfs)]
        print("Weighted, step 2:", y)

    yhat2 = f_neuron(x2)
    y2 = torch.sigmoid(x2)
    print("Outputs, step 2:", yhat2)
    loss2 = loss_fn(yhat2, target=y2)

    opt.zero_grad()
    loss2.backward()
    opt.step()

    print("Params, step 2:", [*f_neuron.parameters()])

    ############################################################################
    # Automatic random sampling starts                                         #
    ############################################################################
    torch.manual_seed(42)
    num_samples = 256

    for k in range(num_samples):
        x = torch.rand([1])
        x = x * x_len + x_left
        y = torch.sigmoid(x)
        yhat = f_neuron(x)
        loss = loss_fn(yhat, target=y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Params, step {2+num_samples}:", [*f_neuron.parameters()])

    show_f_neuron_mfs(f_neuron)
    show_f_neuron(f_neuron)


if __name__ == "__main__":
    main()
