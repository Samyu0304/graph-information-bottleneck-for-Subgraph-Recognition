#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
mkdir -p ../test_results/qed/
mkdir -p ../validate_results/qed/
python main.py