import tensorflow as tf
import model_zoo.callbacks as callbacks
import math


class BaseModel(tf.keras.Model):
    """
    Base Keras Model, you can inherit this Model and
    override '__init__()', 'call()', 'init()', 'callback()' methods
    """

    logger = None

    def __init__(self, config):
        """
        init config, batch_size, epochs
        :param config:
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.batch_size = config.get('batch_size')
        self.epochs = config.get('epochs')
        self.steps_per_epoch = config.get('steps_per_epoch')
        self.validation_steps = config.get('validation_steps')
        self.distributed = config.get('distributed', False)

    def call(self, inputs, training=False, mask=None):
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
        Default call compile method using sgd optimizer and mse loss.
        :return:
        """
        self.compile(optimizer=self.get_optimizer() or self.config.get('optimizer'),
                     loss=self.get_loss() or None,
                     metrics=self.get_metrics() or None,
                     loss_weights=self.get_loss_weights() or None,
                     weighted_metrics=self.get_weighted_metrics() or None,
                     target_tensors=self.get_target_tensors() or None)

    def train(self, train_data, eval_data=None, use_generator=False, **kwargs):
        """
        Train and fit model.
        :param train_data: x, y data pairs for training
        :param eval_data: x, y data pairs for evaluating
        :return: fit result
        """
        if not use_generator:
            # print('Train data', train_data)
            x, y = train_data
            self.init()
            # execute training
            return self.fit(x=x,
                            y=y,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=eval_data,
                            callbacks=self.callbacks())

        if use_generator:
            # use generator
            self.init()

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
            cbs.append(tf.keras.callbacks.EarlyStopping(patience=self.config.get('early_stop_patience')))
        if self.config.get('tensor_board_enable'):
            cbs.append(tf.keras.callbacks.TensorBoard(log_dir=self.config.get('tensor_board_dir', 'events')))
        if self.config.get('checkpoint_enable'):
            cbs.append(callbacks.ModelCheckpoint(
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
                checkpoint_name=self.config.get('checkpoint_name', 'model'),
                checkpoint_restore=self.config.get('checkpoint_restore', True),
                checkpoint_save_freq=self.config.get('checkpoint_save_freq', 2),
                checkpoint_save_best=self.config.get('checkpoint_save_best', True),
                checkpoint_save_latest=self.config.get('checkpoint_save_latest', True),
                checkpoint_save_every=self.config.get('checkpoint_save_every', True),
                checkpoint_save_weights_only=self.config.get('checkpoint_save_weights_only', True)))
        return cbs

    def get_optimizer(self):
        """
        Build optimizer, default to sgd.
        :return:
        """
        return 'sgd'

    def get_loss(self):
        """
        Build loss function, default to `mse`.
        :return:
        """
        return 'mse'

    def get_metrics(self):
        """
        Build metrics
        :return:
        """
        return []

    def get_loss_weights(self):
        """
        Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions
        of different model outputs. The loss value that will be minimized by the model will then be the weighted sum
        of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1
        mapping to the model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.
        :return:
        """
        return None

    def get_weighted_metrics(self):
        """
        List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
        :return:
        """
        return None

    def get_target_tensors(self):
        """
        By default, Keras will create placeholders for the model's target, which will be fed with the target data
        during training. If instead you would like to use your own target tensors (in turn, Keras will not expect
        external Numpy data for these targets at training time), you can specify them via the target_tensors argument.
        It can be a single tensor (for a single-output model), a list of tensors, or a dict mapping output names
        to target tensors.
        :return:
        """
        return None

    def __str__(self):
        """
        To string.
        :return:
        """
        return type(self).__name__
