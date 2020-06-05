FROM tensorflow/tensorflow:1.10.1-gpu

COPY . /model
WORKDIR /model

#ENTRYPOINT ["/bin/bash",\"run_.sh"]
#CMD ["cd","/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/slim/python/slim/nets"]
#RUN git clone "https://github.com/subeans/whitebox.git"
#CMD ["git","clone","https://github.com/subeans/whitebox.git"]
RUN mv whitebox/lenet.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/python/slim/nets
#CMD ["mv","./lenet.py","/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/slim/python/slim/nets"]
#RUN cd /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim
#CMD ["cd","/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/slim/"]
RUN mv /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets.py \
/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim/nets_old.py
#RUN cd ~
#CMD ["mv","nets.py","nets_old.py"]
#CMD ["cd","~"]
RUN mv whitebox/nets.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim
#CMD ["mv","./nets.py","/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/slim"]
#CMD ["cd","/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/slim"]
CMD ["/bin/bash","run_bm.sh"]
