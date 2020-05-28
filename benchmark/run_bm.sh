#!/bin/bash
batchsize=(1 2 4 8 16 32 64 128 256 512)
opt=('sgd' 'rmsprop' 'adadelta' 'adagrad' 'momentum' 'adam' )
for j in {0..5};
  for i in {0..9};
    do python benchmark.py --testVGG16 --no_timeline --iter_benchmark=100 --batchsize=${batchsize[i]} --optimizer=${opt[j]};
done
