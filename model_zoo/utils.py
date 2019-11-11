import json
from os.path import join, exists
import numpy as np
import tensorflow as tf
from os.path import exists
from importlib import import_module


def load_config(config):
    """
    log configs
    :param FLAGS: FLAGS object
    :return: config dict
    """
    loaded = json.load(
        open('%s.json' % join(config.get('checkpoint_dir'), config.get('checkpoint_name')), 'r', encoding='utf-8'))
    for key, value in config.items():
        loaded[key] = value
    return loaded


def load_model(model, checkpoint_dir, checkpoint_name):
    """
    restore model from saved checkpoints
    :param model: model graph
    :return:
    """
    saver = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    specified_checkpoint = join(checkpoint_dir, checkpoint_name)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if specified_checkpoint:
        saver.restore(specified_checkpoint)
        print('Restored from %s' % specified_checkpoint)
    elif latest_checkpoint:
        saver.restore(latest_checkpoint)
        print('Restored from %s' % latest_checkpoint)
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
