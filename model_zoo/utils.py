import json
from os.path import join
from pathlib import Path
import numpy as np
import tensorflow as tf
from importlib import import_module


def load_config(config):
    """
    log configs
    :param FLAGS: FLAGS object
    :return: config dict
    """
    json_path = '%s.json' % join(config.get('checkpoint_dir'), Path(config.get('checkpoint_name')).stem)
    loaded = json.load(open(json_path, 'r', encoding='utf-8'))
    for key, value in config.items():
        loaded[key] = value
    return loaded


def load_model(model, checkpoint_dir, checkpoint_name):
    """
    restore model from saved checkpoints
    :param model: model graph
    :return:
    """
    specified_checkpoint = join(checkpoint_dir, checkpoint_name)
    if specified_checkpoint:
        if '.ckpt' in checkpoint_name:
            model.load_weights(specified_checkpoint)
            print('Restored weights from %s' % specified_checkpoint)
        if '.h5' in checkpoint_name:
            model.load(specified_checkpoint)
            print('Restored all model from %s' % specified_checkpoint)
    else:
        print('No model to restore')


def find_model(model_class_name, model_file_name):
    """
    dynamic find model from model file according to model class name
    :param name: name of model class
    :param name: file name of model class
    :return: model class
    """
    if not model_class_name:
        raise Exception('You must define model_class in flags')
    model_file = import_module(model_file_name)
    model_class = getattr(model_file, model_class_name)
    return model_class


def get_shape(data):
    """
    get shape of data except batch_size
    :param data:
    :return:
    """
    if isinstance(data, np.ndarray) or isinstance(data, tf.Tensor):
        shape = data.shape
        return shape[1:]
    else:
        data = np.asarray(data)
        shape = data.shape
        return shape[1:]
