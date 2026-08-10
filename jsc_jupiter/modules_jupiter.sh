#!/usr/bin/env bash

module --force purge

module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
module load CUDA/12
module load PyTorch/2.5.1