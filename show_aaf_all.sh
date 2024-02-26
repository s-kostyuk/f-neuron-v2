#!/bin/bash

set -e

BASE_SCRIPT="./run_experiment.sh"
#BASE_SCRIPT="echo"
BASE_APP="./post_experiment/show_aaf_individual.py"

COMMON_OPTS=(--net KerasNet --ds CIFAR-10 --opt adam)

echo "---------------- Stage 1 ----------------"
# Not applicable

echo "---------------- Stage 2 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts ReLU \
                --end_ep 200 --tuned \
                --patched \
                "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts ReLU \
                --end_ep 200 --tuned --p24sl \
                --patched \
                "${COMMON_OPTS[@]}"

echo "---------------- Stage 3 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts SiLU \
                --end_ep 200 \
                --patched "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts SiLU \
                --end_ep 200 --p24sl --tuned \
                --patched "${COMMON_OPTS[@]}"

echo "---------------- Stage 4 ----------------"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Tanh --act_cnn ReLU \
               --end_ep 500 --tuned \
               --patched --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts Tanh --act_cnn ReLU \
               --end_ep 500 --tuned --p24sl \
               --patched --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Random --act_cnn ReLU \
               --end_ep 500 --tuned \
               --patched --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"

echo "---------------- Stage 5 ----------------"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Tanh --act_cnn SiLU \
               --end_ep 500 --tuned \
               --patched --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts Tanh --act_cnn SiLU \
               --end_ep 500 --tuned --p24sl \
               --patched --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Random --act_cnn SiLU \
               --end_ep 500 --tuned \
               --patched --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
if false; then
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts Tanh --act_cnn SiLU \
               --end_ep 500 --tuned --p24sl \
               --patched --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
fi

echo "---------------- Stage 6 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts ReLU \
               --end_ep 500 --tuned \
               --patched --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts ReLU \
               --end_ep 500 --tuned --p24sl \
               --patched --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"

echo "---------------- Stage 7 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts SiLU \
               --end_ep 500 --tuned \
               --patched --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts SiLU \
               --end_ep 500 --tuned --p24sl \
               --patched --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"

