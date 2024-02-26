# Deep neural network based on F-neurons and its learning 

Implementation of the experiment as published in the paper (v2, after the peer
review cycle).

## Differences from [v1](https://github.com/s-kostyuk/f-neuron)

- v1: one-time random number generator seeding before starting the set of
  experiment runs; v2: re-seeding the random number generator before each run
  in the set (for each of the argument combinations); re-seeding allows
  reproducing each training run individually without enforcing repetition of the
  whole experiment;
- v1: initial learning rate of 1e-4 with a learning rate decay of 1e-6 per
  minibatch; v2: fixed learning rate of 1e-4 across all epochs; removing LR
  decay simplifies the experiment and its reproduction while having only small
  effect on the final network training result.

## Description of the experiment

The experiment consists of the following steps:

1. Train the base LeNet-5 and KerasNet networks on LeNet-5 and KerasNet datasets
   for 100 epochs using the standard training procedure and RMSprop. Use ReLU as
   the activation function across all network layers.
   
   **NOTE:** Contrary to the experiment in version 1 of the paper (submitted to
   the Soft Computing journal on 2022.09.05), each model variant in v2 is being
   trained from scratch in a clean environment, with the seed value enforced
   individually per run. Hence, v2 of the experiments does not allow a 1-to-1
   reproduction of the results from v1. Nevertheless, the fixed seed value
   allows comparing all network variants in the same starting positions, with
   the same starting weights, and reproducing each part of the experiment
   individually without reproducing the whole set in one run, as it takes a lot
   of time.

2. Train the LeNet-5 and KerasNet networks with ReLU activations in CNN layers
   and Ramp-initialized (2.1), Random-initialized (2.2), and
   Constant-initialized (2.3) F-Neuron activations in the fully connected
   layers. The datasets, number of epochs, the training procedure, optimizers
   are the same as in step 1 (above).

3. (Optional, excluded from the final paper).
   Extend the F-Neuron definition range from `(-1.0;+1.0), 12 MFs` to
   `(-4.0;+4.0), 16 MFs` by using the `fuzzyw` F-neuron variant. Repeat the
   experiment from step 2 with Ramp-initialized (3.1), Random-initialized (3.2),
   and Constant-initialized (3.3) F-Neuron activations in the fully connected
   layers that have the extended definition interval.

4. Repeat steps 1-3 for the following seed values: 85, 100, 128, 1999, 7823.

## Illustrating training with the Sigmoid target

Section 3 of the paper, "The F-neuron activation shape adaptation", includes
step-by-step training of the F-neuron synapse. In order to replicate the
results, execute the [show_mf_ranges.py](./post_experiment/show_mf_ranges.py)
script. Use the [run_experiment.sh](./run_experiment.sh) wrapper if required for setting up the
module search paths and other environment pieces.

## Running experiments

1. NVIDIA GPU recommended with at least 2 GiB of VRAM.
2. Install the requirements from `requirements.txt`.
3. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment variables.
4. Use the root of this repository as the current directory.
5. Add the current directory to `PYTHONPATH` so it can find the modules

This repository contains a wrapper script that sets all the required
environment variables: [run_experiment.sh](./run_experiment.sh). Use the bash shell to
execute the experiment using the wrapper script:

Example:

```shell
user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py  #...
```

## Reproducing results from the paper

Execute the [run_experiment_all.sh](./run_experiment_all.sh) script to perform
all experiments in an automated way. For parallel execution - consult the
list of commands from this script and execute them manually.

Example:

```shell
user@host:~/repo_path$ ./run_experiment_all.sh  # no extra arguments
```

## Visualization of experiment results

Execute the [show_aaf_all.sh](./show_aaf_all.sh) script to visualize all the
adaptive activation functions in an automated way. For parallel execution -
consult the list of commands from this script and execute them manually.

Example:

```shell
user@host:~/repo_path$ ./show_aaf_all.sh  # no extra arguments
```
