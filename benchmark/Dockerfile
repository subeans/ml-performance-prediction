FROM tensorflow/tensorflow:1.10.1-gpu

COPY . /model
WORKDIR /model

#ENTRYPOINT ["/bin/bash",\"run_bm.sh"]
#RUN ls

#CMD ["cd /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets","rm -rf vgg.py"]
#RUN rm -rf vgg.py
#RUN cd ~ 
#RUN rm -rf /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets/vgg.py
CMD ["git clone https://github.com/subeans/whitebox.git"]
#RUN mv /model/whitebox/vgg.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets

#RUN mv /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets.py \
#/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets_old.py

#RUN mv /model/whitebox/nets.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim

CMD ["/bin/bash","run_bm.sh"]

