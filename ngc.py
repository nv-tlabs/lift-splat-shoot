from fire import Fire
import os
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import matplotlib as mpl
import yaml
from time import sleep

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def build():
    cmd = 'docker build . -t lift-splat-shoot-pp'
    print(cmd)
    os.system(cmd)


def run_local():
    cmd = 'nvidia-docker run -v /media/jphilion/52a4e0ca-6b1d-45f6-bb4d-516d3d7b316d1/data/data/nuscenes/:/data/nuscenes --expose=6006 -it --ipc=host lift-splat-shoot-pp'
    print(cmd)
    os.system(cmd)


def push():
    cmd = 'docker tag lift-splat-shoot-pp nvcr.io/nvidian/ct-toronto-ai/lift-splat-shoot-pp'
    print(cmd)
    os.system(cmd)

    cmd = 'docker push nvcr.io/nvidian/ct-toronto-ai/lift-splat-shoot-pp'
    print(cmd)
    os.system(cmd)


def prepare_ngc():
    # the nuscenes dataset is organized strangely on ngc
    # this puts a linked version of the dataset at /workspace/nuscenes
    cmds = """
    mkdir -p /workspace/nuscenes/mini
    mkdir -p /workspace/nuscenes/trainval
    ln -s /mount/data/mini/v1.0-mini /workspace/nuscenes/mini/v1.0-mini
    ln -s /mount/data/mini/maps /workspace/nuscenes/mini/maps
    ln -s /mount/data/trainval/meta/v1.0-trainval /workspace/nuscenes/trainval/v1.0-trainval
    ln -s /mount/data/trainval/meta/maps /workspace/nuscenes/trainval/maps

    # link the raw data
    ln -s /mount/data/mini/samples /workspace/nuscenes/mini/samples
    ln -s /mount/data/trainval/ /workspace/nuscenes/trainval/samples
    ln -s /mount/data/trainval/ /workspace/nuscenes/trainval/sweeps
    
    # linking for adaptation.
    mkdir -p /data/adapt
    ln -s /workspace/nuscenes /data/adapt/nuscenes
    ln -s /mount/carlasim /data/adapt/carlasim
    """
    cmds = cmds.split('\n')
    for cmd in cmds:
        print(cmd)
        os.system(cmd)


def run_ngc(pdata, optimizer='adam'):
    name= f'ml-model.exempt-liftsplatshootpp{pdata}'
    CMD = f"python ngc.py prepare_ngc && python main.py train trainval --pdata={pdata} --optimizer={optimizer} --dataroot=/workspace/nuscenes --num_workers=7 --logdir=/results/runs --gpuid=0 & tensorboard --bind_all --logdir=/results/ --port 6006 --samples_per_plugin='images=1000'"
    cmd = f"""~/ngc batch run --datasetid 40376:/mount/data\
            --image "nvidian/ct-toronto-ai/lift-splat-shoot-pp" --result /results\
            --ace nv-us-west-2 --instance dgx1v.16g.1.norm --port 6006\
            --commandline "{CMD}"\
            -n {name}
    """
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    Fire({
        'build': build,
        'run_local': run_local,
        'push': push,
        'prepare_ngc': prepare_ngc,
        'run_ngc': run_ngc,
    })
