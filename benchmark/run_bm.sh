#!/bin/bash

rm -rf /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets/vgg.py
mv /model/whitebox/vgg.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets
mv /model/whitebox/lenet.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets
mv /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets_old.py
mv whitebox/nets.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim


num=(1 2 3)
batchsize=(1 2 4 8 16 32 64 128 256 512)
opti=(sgd adadelta adam)
#opti=(sgd adadelta adam adagrad rmsprop momentum)
for p in ${opti[*]}; do
    for i in ${batchsize[*]}; do
        for k in ${num[*]}; do
	    #python benchmark.py --testVGGsmall --imgsize=28 --numclasses=10 --no_timeline --iter_benchmark=100 --batchsize=$i --optimizer=$p
    	    python benchmark.py --testLenet --imgsize=28 --numclasses=10 --no_timeline --iter_benchmark=100 --batchsize=$i --optimizer=$p
	    #python benchmark.py --testVGG16 --no_timeline --iter_benchmark=100 --batchsize=$i --optimizer=$p
	done
	
    done
done

touch /results/finish.txt
