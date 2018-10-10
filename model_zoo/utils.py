import json
from os.path import join
import numpy as np
import tensorflow as tf
from os.path import exists
from importlib import import_module

tfe = tf.contrib.eager


def load_config(FLAGS):
    """
    log configs
    :param FLAGS: FLAGS object
    :return: config dict
    """
    config = json.load(open('%s.json' % join(FLAGS.checkpoint_dir, FLAGS.checkpoint_name), 'r', encoding='utf-8'))
    for key, value in FLAGS.flag_values_dict().items():
        config[key] = value
    return config


def load_model(model, checkpoint_dir, checkpoint_name):
    """
    restore model from saved checkpoints
    :param model: model graph
    :return:
    """
    saver = tfe.Saver(model.variables)
    specified_checkpoint = join(checkpoint_dir, checkpoint_name)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if specified_checkpoint and exists(checkpoint_dir) and tf.train.checkpoint_exists(specified_checkpoint):
        saver.restore(specified_checkpoint)
        print('Restored from %s' % specified_checkpoint)
    elif latest_checkpoint and exists(checkpoint_dir) and tf.train.checkpoint_exists(latest_checkpoint):
        saver.restore(latest_checkpoint)
        print('Restored from %s' % latest_checkpoint)
    else:
        print('No model to restore')


def find_model(model_class_name, model_file):
    """
    dynamic find model from model file according to model class name
    :param name: name of model class
    :param name: file name of model class
    :return: model class
    """
    if not model_class_name:
        raise Exception('You must define model_class in flags')
    model_file = import_module(model_file)
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
    
    

