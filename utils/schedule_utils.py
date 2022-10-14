import sys
sys.path.append("/home/v-xiajiao/code/Med-TS-ExpertCL/models")
import os, gc
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.sparse
import yaml
import json
import inspect
import importlib
import omegaconf
from matplotlib.ticker import MaxNLocator


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

def load_config_file(file_path):
    yaml.add_constructor('!join', join)
    with open(file_path, 'r') as fp:
        # return OmegaConf.load(fp)
        return yaml.unsafe_load(fp)

def load_model(name, args):
    try:
        Model = getattr(importlib.import_module(
            name, package=__package__), name)
    except:
        raise ValueError(
            f'Invalid Module File Name or Invalid Class Name {name}.{name}!')
    model = instancialize(Model, args)
    return model

def instancialize(Model, args, **other_args):
    """ Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.hparams.
    """
    args = vars(args)
    if '_content' in args:
        args = args['_content']
    class_args = inspect.getargspec(Model.__init__).args[1:]
    inkeys = args.keys()
    args1 = {}
    for arg in class_args:
        if arg in inkeys:
            if type(args[arg]) == omegaconf.nodes.AnyNode:
                args1[arg] = args[arg]._val
            else:
                args1[arg] = args[arg]
    args1.update(other_args)
    return Model(**args1)
