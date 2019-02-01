import tensorflow as tf
import model_zoo.callbacks as callbacks
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K

tfe = tf.contrib.eager


class BaseModel(tf.keras.Model):
    """
    Base Keras Model, you can inherit this Model and
    override '__init__()', 'call()', 'init()', 'callback()' methods
    """
    
    def __init__(self, config):
        """
        init config, batch_size, epochs
        :param config:
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.steps_per_epoch = config['steps_per_epoch']
        self.validation_steps = config['validation_steps']
    
    def set_inputs(self, inputs):
        """
        set inputs and output shape according to inputs
        :param inputs: inputs data or data piece
        :return:
        """
        if isinstance(inputs, (list, tuple)):
            if tensor_util.is_tensor(inputs[0]):
                dummy_output_values = self.call(
                    training_utils.cast_if_floating_dtype(inputs[:1]))
            else:
                dummy_output_values = self.call(
                    [ops.convert_to_tensor(v, dtype=K.floatx()) for v in inputs[:1]])
            dummy_input_values = list(inputs[:1])
        else:
            if tensor_util.is_tensor(inputs):
                dummy_output_values = self.call(
                    training_utils.cast_if_floating_dtype(inputs[:1]))
            else:
                dummy_output_values = self.call(
                    ops.convert_to_tensor(inputs[:1], dtype=K.floatx()))
            dummy_input_values = [inputs[:1]]
        if isinstance(dummy_output_values, (list, tuple)):
            dummy_output_values = list(dummy_output_values)
        else:
            dummy_output_values = [dummy_output_values]
        self.outputs = [
            base_layer.DeferredTensor(shape=(None for _ in v.shape),
                                      dtype=v.dtype) for v in dummy_output_values]
        self.inputs = [
            base_layer.DeferredTensor(shape=(None for _ in v.shape),
                                      dtype=v.dtype) for v in dummy_input_values]
        self.input_names = [
            'input_%d' % (i + 1) for i in range(len(dummy_input_values))]
        self.output_names = [
            'output_%d' % (i + 1) for i in range(len(dummy_output_values))]
        self.built = True
        self.init()
    
    def call(self, inputs, training=None, mask=None):
        """
        build your models
        :param inputs: inputs x
        :param training: training flag
        :param mask: mask flag
        :return: y_pred
        """
        return inputs
    
    def init(self):
        """
        default call compile method using sgd optimizer and mse loss
        :return:
        """
        self.compile(optimizer=self.optimizer(), loss='mse')
    
    def train(self, train_data, eval_data=None, use_generator=False, **kwargs):
        """
        train and fit model
        :param train_data: x, y data pairs for training
        :param eval_data: x, y data pairs for evaluating
        :return: fit result
        """
        print('Training...')
        if not use_generator:
            x, y = train_data
            self.set_inputs(x)
            return self.fit(x=x,
                            y=y,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=eval_data,
                            callbacks=self.callbacks())
        # use generator
        x, y = next(train_data)
        self.set_inputs(x)
        
        # get train size, eval size
        train_size = kwargs.get('train_size', 0)
        eval_size = kwargs.get('eval_size', 0)
        
        # calculate steps_per_epoch
        steps_per_epoch = self.steps_per_epoch
        if not steps_per_epoch and train_size:
            steps_per_epoch = int(train_size / self.batch_size + 1)
        if not steps_per_epoch:
            raise Exception('You must specify `steps_per_epoch` argument if `train_size` is not set')
        
        # calculate validation steps
        validation_steps = self.validation_steps
        if not validation_steps and eval_size:
            validation_steps = int(eval_size / self.batch_size + 1)
        if not validation_steps:
            validation_steps = 1
        
        # fit generator
        return self.fit_generator(train_data,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=self.epochs,
        
                                  validation_data=eval_data,
                                  validation_steps=validation_steps,
                                  callbacks=self.callbacks()
                                  )
    
    def infer(self, test_data, batch_size=None):
        """
        do inference, default call predict method
        :param test_data: x data
        :param batch_size: batch_size
        :return:
        """
        x = test_data
        if not batch_size:
            batch_size = self.batch_size
        return self.predict(x, batch_size)
    
    def callbacks(self):
        """
        default callbacks, including logger, early stop, tensor board, checkpoint
        :return:
        """
        cbs = []
        cbs.append(tf.keras.callbacks.BaseLogger())
        if self.config.get('early_stop_enable'):
            cbs.append(tf.keras.callbacks.EarlyStopping(patience=self.config.get('early_stop_patience', 50)))
        if self.config.get('tensor_board_enable'):
            cbs.append(tf.keras.callbacks.TensorBoard(log_dir=self.config.get('tensor_board_dir', 'events')))
        if self.config.get('checkpoint_enable'):
            cbs.append(callbacks.ModelCheckpoint(
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
                checkpoint_name=self.config.get('checkpoint_name', 'model'),
                checkpoint_restore=self.config.get('checkpoint_restore', True),
                checkpoint_save_freq=self.config.get('checkpoint_save_freq', 2),
                checkpoint_save_best_only=self.config.get('checkpoint_save_best_only', False)))
        return cbs
    
    def optimizer(self):
        """
        build optimizer, default to sgd and lr 0.01
        :return:
        """
        return tf.train.GradientDescentOptimizer(self.config.get('learning_rate', 0.01))
