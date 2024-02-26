import time
from typing import Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader

from adaptive_afs.fuzzy import FNeuronAct
from adaptive_afs.cont.leaf import LEAF
from post_experiment.show_af_diff import estimate_error
from misc import RunningStat


class TanhSampler(IterableDataset):
    def __init__(self, bs=32, dim=(10,), left=-5.0, right=+5.0):
        super().__init__()
        self._bs = bs
        self._dim = dim
        self._left = left
        self._right = right
        self._ds_size = 16386

    def __iter__(self):
        return self._sample_iter()

    def _sample_iter(self):
        xs, ys = self._get_sample()
        for x, y in zip(xs, ys):
            yield x, y

    def _get_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(
            size=(self._ds_size, *self._dim), requires_grad=False
        )
        x = x * (self._right - self._left) + self._left
        y = torch.tanh(x)
        return x, y


def main():
    torch.set_default_tensor_type(torch.DoubleTensor)
    input_dim = (1,)
    test_nfn = True

    if test_nfn:
        act = FNeuronAct(
            left=-5.0, right=+5.0, count=512,
            #init_f=FNeuronAct.get_init_f_by_name('Random'),
            init_f=FNeuronAct.get_init_f_by_name('Tanh'),
            input_dim=input_dim
        )
    else:
        act = LEAF(size=input_dim, init_as='Tanh')

    params_orig = torch.stack([p.detach() for p in act.parameters()])

    ds = DataLoader(
        TanhSampler(dim=input_dim, left=-6.0, right=+6.0),
        batch_size=512
    )
    opt = torch.optim.SGD(act.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    err_pre = estimate_error(torch.tanh, act, left=-14.0, right=+14.0)
    print('ERR before training:', err_pre)

    ep = None
    last_loss = None

    ep_time_stat = RunningStat()

    try:
        for ep in range(100):
            ep_start = time.time()
            for mb in ds:
                x, y = mb
                y_hat = act(x)
                loss = loss_fn(y_hat, target=y)
                last_loss = loss.item()

                opt.zero_grad()
                loss.backward()
                opt.step()

            ep_end = time.time()
            ep_time = ep_end - ep_start
            ep_time_stat.push(ep_time)
            print(ep, last_loss, ep_time)

    except KeyboardInterrupt:
        print(f'Interrupted at ep {ep}, loss {last_loss}')

    print('EP TIME mean, stddev:', ep_time_stat.mean, ep_time_stat.stddev)

    err_post = estimate_error(torch.tanh, act, left=-14.0, right=+14.0)
    print('ERR after training:', err_post)

    params_new = torch.stack([p.detach() for p in act.parameters()])
    params_diff = params_new - params_orig
    print('Change in params:', params_diff)


if __name__ == "__main__":
    main()
