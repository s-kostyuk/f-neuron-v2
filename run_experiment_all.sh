#!/bin/bash

set -e

BASE_SCRIPT="./run_experiment.sh"
#BASE_SCRIPT="echo"
BASE_APP="./experiments/train_individual.py"

if [[ -n "$1" ]]; then
  SEED="$1"
else
  SEED=42
fi

COMMON_OPTS=(--opt rmsprop --seed "$SEED" --bs 64 --dev gpu)
COMMON_OPTS+=("--wandb")

function train_nets() {
  local _OPTS

  _OPTS=("$@")

  echo "---------------- Stage 1 ----------------"
  "$BASE_SCRIPT" "$BASE_APP" base --act_cnn ReLU --acts ReLU \
    --start_ep 0 --end_ep 100 "${_OPTS[@]}"

  echo "--------------- Stage 2.1 ---------------"
  "$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn --act_cnn ReLU --acts Ramp \
    --start_ep 0 --end_ep 100 "${_OPTS[@]}"

  echo "--------------- Stage 2.2 ---------------"
  "$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn --act_cnn ReLU --acts Random \
    --start_ep 0 --end_ep 100 "${_OPTS[@]}"

  echo "--------------- Stage 2.3 ---------------"
  "$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn --act_cnn ReLU --acts Constant \
    --start_ep 0 --end_ep 100 "${_OPTS[@]}"

  echo "--------------- Stage 3.1 ---------------"
  "$BASE_SCRIPT" "$BASE_APP" fuzzyw_ffn --act_cnn ReLU --acts Ramp \
    --start_ep 0 --end_ep 100 "${_OPTS[@]}"

  echo "--------------- Stage 3.2 ---------------"
  "$BASE_SCRIPT" "$BASE_APP" fuzzyw_ffn --act_cnn ReLU --acts Random \
    --start_ep 0 --end_ep 100 "${_OPTS[@]}"

  echo "--------------- Stage 3.3 ---------------"
  "$BASE_SCRIPT" "$BASE_APP" fuzzyw_ffn --act_cnn ReLU --acts Constant \
    --start_ep 0 --end_ep 100 "${_OPTS[@]}"

  return 0
}

echo "============ LeNet-5 F-MNIST ============"
train_nets --net LeNet-5 --ds F-MNIST "${COMMON_OPTS[@]}"

echo "============ LeNet-5 CIFAR-10 ==========="
train_nets --net LeNet-5 --ds CIFAR-10 "${COMMON_OPTS[@]}"

echo "=========== KerasNet F-MNIST ============"
train_nets --net KerasNet --ds F-MNIST "${COMMON_OPTS[@]}"

echo "=========== KerasNet CIFAR-10 ==========="
train_nets --net KerasNet --ds CIFAR-10 "${COMMON_OPTS[@]}"