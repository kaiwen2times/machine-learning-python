# ==================================================================
# modules for machine learning on CPU
# ------------------------------------------------------------------
# python        3.6    (apt)
# torch         latest (git)
# chainer       latest (pip)
# cntk          2.3    (pip)
# jupyter       latest (pip)
# mxnet         latest (pip)
# pytorch       0.3.0  (pip)
# tensorflow    latest (pip)
# theano        latest (git)
# keras         latest (pip)
# lasagne       latest (git)
# opencv        latest (git)
# sonnet        latest (pip)
# caffe         latest (git)
# ==================================================================

FROM nvidia/cuda:8.0-cudnn6-devel
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        scikit-learn \
        matplotlib \
        Cython \
        && \

# ==================================================================
# torch
# ------------------------------------------------------------------

    $GIT_CLONE https://github.com/torch/distro.git ~/torch --recursive && \

    cd ~/torch/exe/luajit-rocks && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_LUAJIT21=ON \
          .. && \
    make -j"$(nproc)" install && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libreadline-dev \
        && \

    $GIT_CLONE https://github.com/Yonaba/Moses ~/moses && \
    cd ~/moses && \
    luarocks install rockspec/moses-1.6.1-1.rockspec && \

    cd ~/torch && \
    sed -i 's/extra\/cudnn/extra\/cudnn \&\& git checkout R6/' install.sh && \
    sed -i 's/$PREFIX\/bin\/luarocks/luarocks/' install.sh && \
    sed -i '/qt/d' install.sh && \
    sed -i '/Installing Lua/,/^cd \.\.$/d' install.sh && \
    sed -i '/path_to_nvidiasmi/,/^fi$/d' install.sh && \
    sed -i '/Restore anaconda/,/^Not updating$/d' install.sh && \
    sed -i '/You might want to/,/^fi$/d' install.sh && \
    sed -i 's/\[ -x "$path_to_nvcc" \]/false/' install.sh && \
    yes no | ./install.sh && \

# ==================================================================
# boost
# ------------------------------------------------------------------

    wget -O ~/boost.tar.gz https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz && \
    tar -zxf ~/boost.tar.gz -C ~ && \
    cd ~/boost_* && \
    ./bootstrap.sh --with-python=python3.6 && \
    ./b2 install --prefix=/usr/local && \

# ==================================================================
# chainer
# ------------------------------------------------------------------

    $PIP_INSTALL \
        chainer \
        && \

# ==================================================================
# cntk
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        openmpi-bin \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        && \

    $PIP_INSTALL \
        https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3-cp36-cp36m-linux_x86_64.whl \
        && \

# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyter \
        && \

# ==================================================================
# mxnet
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libatlas-base-dev \
        graphviz \
        && \

    $PIP_INSTALL \
        mxnet \
        graphviz \
        && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl \
        torchvision \
        && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

    $PIP_INSTALL \
        tensorflow \
        && \

# ==================================================================
# theano
# ------------------------------------------------------------------

    $GIT_CLONE https://github.com/Theano/Theano ~/theano && \
    cd ~/theano && \
    $PIP_INSTALL \
        . && \

# ==================================================================
# keras
# ------------------------------------------------------------------

    $PIP_INSTALL \
        h5py \
        keras \
        && \

# ==================================================================
# lasagne
# ------------------------------------------------------------------

    $GIT_CLONE https://github.com/Lasagne/Lasagne ~/lasagne && \
    cd ~/lasagne && \
    $PIP_INSTALL \
        . && \

# ==================================================================
# opencv
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        && \

    $GIT_CLONE https://github.com/opencv/opencv ~/opencv && \
    mkdir -p ~/opencv/build && cd ~/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          .. && \
    make -j"$(nproc)" install && \

# ==================================================================
# sonnet
# ------------------------------------------------------------------

    $PIP_INSTALL \
        dm-sonnet \
        && \

# ==================================================================
# caffe
# ------------------------------------------------------------------

    $GIT_CLONE https://github.com/BVLC/caffe ~/caffe && \
    cp ~/caffe/Makefile.config.example ~/caffe/Makefile.config && \
    sed -i 's/# CPU_ONLY/CPU_ONLY/g' ~/caffe/Makefile.config && \
    sed -i 's/# PYTHON_LIBRARIES/PYTHON_LIBRARIES/g' ~/caffe/Makefile.config && \
    sed -i 's/# WITH_PYTHON_LAYER/WITH_PYTHON_LAYER/g' ~/caffe/Makefile.config && \
    sed -i 's/# OPENCV_VERSION/OPENCV_VERSION/g' ~/caffe/Makefile.config && \
    sed -i 's/2\.7/3\.6/g' ~/caffe/Makefile.config && \
    sed -i 's/3\.5/3\.6/g' ~/caffe/Makefile.config && \
    sed -i 's/\/usr\/lib\/python/\/usr\/local\/lib\/python/g' ~/caffe/Makefile.config && \
    sed -i 's/\/usr\/local\/include/\/usr\/local\/include \/usr\/include\/hdf5\/serial/g' ~/caffe/Makefile.config && \
    sed -i 's/hdf5/hdf5_serial/g' ~/caffe/Makefile && \
    cd ~/caffe && \
    make -j"$(nproc)" -Wno-deprecated-gpu-targets distribute && \

    # fix ValueError caused by python-dateutil 1.x
    sed -i 's/,<2//g' ~/caffe/python/requirements.txt && \

    $PIP_INSTALL \
        -r ~/caffe/python/requirements.txt && \

    cd ~/caffe/distribute/bin && \
    for file in *.bin; do mv "$file" "${file%%.bin}"; done && \
    cd ~/caffe/distribute && \
    cp -r bin include lib proto /usr/local/ && \
    cp -r python/caffe /usr/local/lib/python3.6/dist-packages/ && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 8888 6006