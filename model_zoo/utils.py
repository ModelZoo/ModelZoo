import json
from os.path import join
import numpy as np
import tensorflow as tf
from os.path import exists

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


def get_shape(test_data):
    """
    get shape of data except batch_size
    :param test_data:
    :return:
    """
    if isinstance(test_data, np.ndarray) or isinstance(test_data, tf.Tensor):
        shape = test_data.shape
        return shape[1:]
    else:
        data = np.asarray(test_data)
        shape = data.shape
        return shape[1:]
