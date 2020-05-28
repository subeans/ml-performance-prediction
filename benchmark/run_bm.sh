#!/bin/bash
batchsize=(1 2 4 8 16 32 64 128 256 512)

for i in {0..9};
    do python benchmark.py --testVGG16 --no_timeline --iter_benchmark=100 --batchsize=${batchsize[i]} --optimizer=sgd;
done
