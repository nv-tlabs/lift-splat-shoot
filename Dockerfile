FROM nvcr.io/nvidian/ct-toronto-ai/pytorch1.2:0.1
MAINTAINER jphilion@nvidia.com

RUN apt-get update

RUN pip uninstall -y torchvision torch tensorboardX
RUN pip install fire seaborn torchvision
RUN pip install --no-deps tensorboardX
RUN pip install PyYAML --ignore-installed PyYAML
#RUN pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
RUN pip install --no-dependencies efficientnet-pytorch==0.7.0 nuscenes-devkit==1.0.6
RUN pip install pyquaternion cachetools descartes shapely ipdb
RUN pip install --no-deps kornia==0.2
RUN pip install lyft_dataset_sdk
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch_lightning
RUN pip install wandb --upgrade

# spconv
RUN apt-get remove -y cmake && pip install --upgrade cmake
RUN apt-get update
RUN apt-get install -y software-properties-common && apt-get update
RUN add-apt-repository universe && apt-get update
RUN apt-get install -y libboost-all-dev tree
RUN git clone https://github.com/traveller59/spconv.git --recursive
RUN cd spconv && python setup.py bdist_wheel && pip install ./dist/*.whl


WORKDIR /workspace
COPY . lift-splat-shoot-pp
WORKDIR lift-splat-shoot-pp
