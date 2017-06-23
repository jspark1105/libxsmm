#!/bin/bash

icc  -xMIC-AVX512 -qno-offload -fpic -Wall -diag-disable 1879,3415,10006,10010,10411,13003 -O3 -fno-alias -ansi-alias -qoverride_limits -fp-model fast=2 -fopenmp -pthread -DLIBXSMM_BUILD -DNDEBUG -D__STATIC=1 -D_REENTRANT -Iinclude -Ibuild -I/scratch/jpark103/packages/libxsmm/src -c /scratch/jpark103/packages/libxsmm/src/libxsmm_dnn_convolution_winograd_forward.c -S -fsource-asm
