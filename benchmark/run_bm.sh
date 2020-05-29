#!/bin/bash
batchsize=(1 2 4 8 16 32 64 128 256 512)
opti=(sgd adam adadelta adagrad rmsprop momentum)
for p in ${opti[*]}; do
    for i in ${batchsize[*]}; do
    	python benchmark.py --testVGG16 --no_timeline --iter_benchmark=100 --batchsize=$i --optimizer=$p
	done
done
