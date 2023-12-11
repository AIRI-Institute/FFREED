FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

USER root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
USER user

COPY environment.yml app/environment.yml
COPY config.h.cmake app/config.h.cmake
USER root
RUN chmod 775 app
RUN chown -R user:user app
USER user

RUN conda env update --file app/environment.yml --prune

RUN cd app &&\
    git clone https://github.com/molecularsets/moses.git && \
    cd moses && \
    pip install . && \
    cd .. && \
    rm -rf moses

RUN cd app && \
    git clone https://github.com/e-kwsm/openbabel.git && \
    cd openbabel && \
    git checkout mt19937_64 && \
    mv ../config.h.cmake src/config.h.cmake && \
    mkdir ob-build && \
    cd ob-build && \
    cmake -DRUN_SWIG=ON -DPYTHON_BINDINGS=ON ..
USER root
RUN cd app/openbabel/ob-build && \
    make install
USER user

RUN conda clean -ya
RUN rm -rf app
