import json
from os.path import join
import numpy as np
import tensorflow as tf
from os.path import exists
from importlib import import_module
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine.training_utils import standardize_single_array

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


def standardize_input_data(data,
                           names,
                           shapes=None,
                           check_batch_axis=True,
                           exception_prefix=''):
    """Normalizes inputs and targets provided by users.
  
    Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network's expectations.
  
    Arguments:
        data: User-provided input data (polymorphic).
        names: List of expected array names.
        shapes: Optional list of expected array shapes.
        check_batch_axis: Boolean; whether to check that
            the batch axis of the arrays matches the expected
            value found in `shapes`.
        exception_prefix: String prefix used for exception formatting.
  
    Returns:
        List of standardized input arrays (one array per model input).
  
    Raises:
        ValueError: in case of improperly formatted user-provided data.
    """
    if not names:
        if (data is not None and hasattr(data, '__len__') and len(data) and
            not isinstance(data, dict)):
            raise ValueError('Error when checking model ' + exception_prefix + ': '
                                                                               'expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]
    
    if isinstance(data, dict):
        try:
            data = [
                data[x].values
                if data[x].__class__.__name__ == 'DataFrame' else data[x]
                for x in names
            ]
        except KeyError as e:
            raise ValueError('No data provided for "' + e.args[0] + '". Need data '
                                                                    'for each key in: ' + str(names))
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], (list, tuple)):
            data = [np.asarray(d) for d in data]
        elif len(names) == 1 and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            data = [
                x.values if x.__class__.__name__ == 'DataFrame' else x for x in data
            ]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
    data = [standardize_single_array(x) for x in data]
    
    # if len(data) != len(names):
    #   if data and hasattr(data[0], 'shape'):
    #     raise ValueError('Error when checking model ' + exception_prefix +
    #                      ': the list of Numpy arrays that you are passing to '
    #                      'your model is not the size the model expected. '
    #                      'Expected to see ' + str(len(names)) + ' array(s), '
    #                      'but instead got the following list of ' +
    #                      str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
    #   elif len(names) > 1:
    #     raise ValueError(
    #         'Error when checking model ' + exception_prefix +
    #         ': you are passing a list as input to your model, '
    #         'but the model expects a list of ' + str(len(names)) +
    #         ' Numpy arrays instead. The list you passed was: ' + str(data)[:200])
    #   elif len(data) == 1 and not hasattr(data[0], 'shape'):
    #     raise TypeError('Error when checking model ' + exception_prefix +
    #                     ': data should be a Numpy array, or list/dict of '
    #                     'Numpy arrays. Found: ' + str(data)[:200] + '...')
    #   elif len(names) == 1:
    #     data = [np.asarray(data)]
    
    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None:
                if tensor_util.is_tensor(data[i]):
                    tensorshape = data[i].get_shape()
                    if not tensorshape:
                        continue
                    data_shape = tuple(tensorshape.as_list())
                else:
                    data_shape = data[i].shape
                shape = shapes[i]
                if len(data_shape) != len(shape):
                    raise ValueError('Error when checking ' + exception_prefix +
                                     ': expected ' + names[i] + ' to have ' +
                                     str(len(shape)) + ' dimensions, but got array '
                                                       'with shape ' + str(data_shape))
                if not check_batch_axis:
                    data_shape = data_shape[1:]
                    shape = shape[1:]
                for dim, ref_dim in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim is not None and dim is not None:
                        raise ValueError(
                            'Error when checking ' + exception_prefix + ': expected ' +
                            names[i] + ' to have shape ' + str(shape) +
                            ' but got array with shape ' + str(data_shape))
    return data
