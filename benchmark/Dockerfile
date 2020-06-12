FROM tensorflow/tensorflow:1.10.1-gpu

COPY . /model
WORKDIR /model

#ENTRYPOINT ["/bin/bash",\"run_bm.sh"]
RUN ls
RUN mv ~/whitebox/lenet.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets

RUN mv /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets.py \
/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets_old.py

RUN mv /whitebox/nets.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim

CMD ["/bin/bash","run_bm.sh"]
