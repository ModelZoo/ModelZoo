import types
from copy import copy

import tensorflow as tf
import model_zoo.callbacks as callbacks
import math


class Framework(object):
    """
    Base framework of model.
    """
    logger = None
    
    def __init__(self, config):
        """
        init config, batch_size, epochs.
        :param config:
        """
        self.model = None
        self.config = config
        self.batch_size = config.get('batch_size')
        self.epochs = config.get('epochs')
        self.steps_per_epoch = config.get('steps_per_epoch')
        self.validation_steps = config.get('validation_steps')
        self.distributed = config.get('distributed', False)
        
        # init model
        self.init()
    
    def inputs(self):
        """
        You should implements inputs.
        :return:
        """
        return NotImplementedError
    
    def outputs(self, inputs):
        """
        Build model using inputs
        :param inputs:
        :return:
        """
        return inputs
    
    def init(self):
        """
        Default call compile method using sgd optimizer and mse loss.
        :return:
        """
        inputs = self.inputs()
        outputs = self.outputs(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer() or self.config.get('optimizer'),
                           loss=self.loss() or None,
                           metrics=self.metrics() or None,
                           loss_weights=self.loss_weights() or None,
                           weighted_metrics=self.weighted_metrics() or None,
                           target_tensors=self.target_tensors() or None)
        if self.config.get('debug'):
            self.model.summary()
    
    def train(self, x, y=None, **kwargs):
        """
        Train and fit model.
        :param train_data: x, y data pairs for training
        :param eval_data: x, y data pairs for evaluating
        :return: fit result
        """
        # fit for generator
        if isinstance(x, types.GeneratorType):
            # get train size, eval size
            train_size = kwargs.get('train_size', 0)
            eval_size = kwargs.get('eval_size', 0)
            
            # calculate steps_per_epoch
            steps_per_epoch = self.steps_per_epoch
            if not steps_per_epoch and train_size:
                steps_per_epoch = math.ceil(train_size / self.batch_size)
            if not steps_per_epoch:
                raise Exception('You must specify `steps_per_epoch` argument if `train_size` is not set')
            
            # calculate validation steps
            validation_steps = self.validation_steps
            if not validation_steps and eval_size:
                validation_steps = math.ceil(eval_size / self.batch_size)
            if not validation_steps:
                validation_steps = 1
            
            args = ['x', 'y', 'batch_size', 'epochs', 'verbose', 'steps_per_epoch',
                    'callbacks', 'validation_split', 'validation_data', 'validation_steps',
                    'validation_freq', 'class_weight', 'max_queue_size', 'sample_weight',
                    'workers', 'use_multiprocessing', 'shuffle', 'initial_epoch']
            keys = copy(list(kwargs.keys()))
            for key in keys:
                if key not in args: kwargs.pop(key)
            
            # fit generator
            return self.model.fit_generator(x,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=self.epochs,
                                            validation_data=kwargs.pop('validation_data'),
                                            validation_steps=validation_steps,
                                            callbacks=self.callbacks(),
                                            **kwargs)
        # fit for normal data
        else:
            args = ['generator', 'steps_per_epoch', 'epochs', 'verbose',
                    'callbacks', 'validation_data', 'validation_steps',
                    'validation_freq', 'class_weight', 'max_queue_size',
                    'workers', 'use_multiprocessing', 'shuffle', 'initial_epoch']
            keys = copy(list(kwargs.keys()))
            for key in keys:
                if key not in args: kwargs.pop(key)
            # execute training
            return self.model.fit(x=x,
                                  y=y,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  validation_data=kwargs.pop('validation_data'),
                                  callbacks=self.callbacks(),
                                  **kwargs)
    
    def infer(self, x, batch_size=None, **kwargs):
        """
        Do inference, default call predict method.
        :param test_data: x data
        :param batch_size: batch_size
        :return:
        """
        if not batch_size:
            batch_size = self.batch_size
        return self.model.predict(x, batch_size, **kwargs)
    
    def evaluate(self, x, y, batch_size=None, **kwargs):
        """
        Do evaluate, default call evaluate method.
        :param x:
        :param y:
        :param batch_size:
        :param kwargs:
        :return:
        """
        if not batch_size:
            batch_size = self.batch_size
        return self.model.evaluate(x, y, batch_size, **kwargs)
    
    def callbacks(self):
        """
        default callbacks, including logger, early stop, tensor board, checkpoint
        :return:
        """
        cbs = []
        cbs.append(tf.keras.callbacks.BaseLogger())
        if self.config.get('early_stop_enable'):
            cbs.append(tf.keras.callbacks.EarlyStopping(patience=self.config.get('early_stop_patience')))
        if self.config.get('tensor_board_enable'):
            cbs.append(tf.keras.callbacks.TensorBoard(log_dir=self.config.get('tensor_board_dir', 'events')))
        if self.config.get('checkpoint_enable'):
            cbs.append(callbacks.ModelCheckpoint(
                checkpoint_config=self.config,
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
                checkpoint_name=self.config.get('checkpoint_name', 'model'),
                checkpoint_restore=self.config.get('checkpoint_restore', True),
                checkpoint_save_freq=self.config.get('checkpoint_save_freq', 2),
                checkpoint_save_best=self.config.get('checkpoint_save_best', True),
                checkpoint_save_latest=self.config.get('checkpoint_save_latest', True),
                checkpoint_save_every=self.config.get('checkpoint_save_every', True),
                checkpoint_save_weights_only=self.config.get('checkpoint_save_weights_only', True)))
        return cbs
    
    def optimizer(self):
        """
        Build optimizer, default to sgd.
        :return:
        """
        return 'sgd'
    
    def loss(self):
        """
        Build loss function, default to `mse`.
        :return:
        """
        return 'mse'
    
    def metrics(self):
        """
        Build metrics
        :return:
        """
        return []
    
    def loss_weights(self):
        """
        Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions
        of different model outputs. The loss value that will be minimized by the model will then be the weighted sum
        of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1
        mapping to the model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.
        :return:
        """
        return None
    
    def weighted_metrics(self):
        """
        List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
        :return:
        """
        return None
    
    def target_tensors(self):
        """
        By default, Keras will create placeholders for the model's target, which will be fed with the target data
        during training. If instead you would like to use your own target tensors (in turn, Keras will not expect
        external Numpy data for these targets at training time), you can specify them via the target_tensors argument.
        It can be a single tensor (for a single-output model), a list of tensors, or a dict mapping output names
        to target tensors.
        :return:
        """
        return None
