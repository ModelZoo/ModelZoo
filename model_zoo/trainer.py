from __future__ import absolute_import, division, print_function, unicode_literals
import types
from model_zoo.exceptions import DefineException
from model_zoo.logger import get_logger
from model_zoo.utils import find_model
from model_zoo import flags
import math

# ========== Base Configs =================
flags.DEFINE_integer('batch_size', 32, help='Batch size', allow_override=True)
flags.DEFINE_float('learning_rate', 0.01, help='Learning rate', allow_override=True)
flags.DEFINE_integer('epochs', 100, help='Max epochs', allow_override=True)
flags.DEFINE_integer('validation_steps', 1, help='Validation steps', allow_override=True)
flags.DEFINE_integer('steps_per_epoch', 0, help='Steps per epoch while using generator', allow_override=True)
flags.DEFINE_string('optimizer', 'rmsprop', help='Default Optimizer', allow_override=True)

# ========== Early Stop Configs ================
flags.DEFINE_bool('early_stop_enable', True, help='Whether to enable early stop', allow_override=True)
flags.DEFINE_integer('early_stop_patience', 20, help='Early stop patience', allow_override=True)

# ========== Checkpoint Configs =================
flags.DEFINE_bool('checkpoint_enable', True, help='Whether to save model checkpoint', allow_override=True)
flags.DEFINE_string('checkpoint_dir', 'checkpoints', help='Data source dir', allow_override=True)
flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name', allow_override=True)
flags.DEFINE_bool('checkpoint_restore', False, help='Model restore', allow_override=True)
flags.DEFINE_integer('checkpoint_save_freq', 2, help='Save model every epoch number', allow_override=True)
flags.DEFINE_bool('checkpoint_save_weights_only', True, help='Save weights of model only', allow_override=True)
flags.DEFINE_bool('checkpoint_save_best', True, help='Save best model', allow_override=True)
flags.DEFINE_bool('checkpoint_save_every', True, help='Save every model', allow_override=True)
flags.DEFINE_bool('checkpoint_save_latest', True, help='Save latest model', allow_override=True)

# ========== Log System ================
flags.DEFINE_bool('log_enable', True, help='Whether to enable Log System', allow_override=True)
flags.DEFINE_string('log_level', 'DEBUG', help='Log Level', allow_override=True)
flags.DEFINE_string('log_rotation', '100MB', help='Log file rotation', allow_override=True)
flags.DEFINE_string('log_retention', None, help='Log file retention', allow_override=True)
flags.DEFINE_string('log_format', '{time} - {level} - {module} - {file} - {message}', help='Log record format',
                    allow_override=True)
flags.DEFINE_string('log_folder', './logs/', help='Folder of log file', allow_override=True)
flags.DEFINE_string('log_file', 'train.log', help='Name of log file', allow_override=True)
flags.DEFINE_string('log_path', '', help='File path of log file', allow_override=True)

# ========== TensorBoard Events Configs =================
flags.DEFINE_bool('tensor_board_enable', True, help='Whether to enable TensorBoard events', allow_override=True)
flags.DEFINE_string('tensor_board_dir', 'events', help='TensorBoard events dir', allow_override=True)

# ========== Other Basic Configs ==================
flags.DEFINE_string('model_file_name', 'models', help='Path of model file which including model class',
                    allow_override=True)
flags.DEFINE_string('model_class_name', 'Model', help='Model class name, default to Model',
                    allow_override=True)


class BaseTrainer(object):
    """
    Base Trainer using eager mode, you can override tf.flags to define new config
    and you need to implement prepare_data method to prepare train_data and eval_data
    """

    def __init__(self):
        """
        You need to define model_class in your Trainer.
        """
        self.config = flags.FLAGS.flag_values_dict()

        # get logger
        logger = get_logger(self.config)
        self.logger = logger
        self.logger.debug(f'training config {self.config}')

        # init model class
        model_class_name, model_file_name = self.config.get('model_class_name'), self.config.get('model_file_name')
        self.model_class = find_model(model_class_name, model_file_name)
        self.logger.debug(f'model class {self.model_class} found')

        checkpoint_name = self.config.get('checkpoint_name')
        checkpoint_save_weights_only = self.config.get('checkpoint_save_weights_only')

        if checkpoint_save_weights_only and not '.ckpt' in checkpoint_name:
            raise DefineException(
                'you must specify `.ckpt` in your checkpoint name while `checkpoint_save_weights_only` is True')

        if not checkpoint_save_weights_only and not '.h5' in checkpoint_name:
            raise DefineException(
                'you must specify `.h5` in your checkpoint name while `checkpoint_save_weights_only` is False')

    def prepare_data(self):
        """
        You need to implement this method.
        :return:
        """
        raise NotImplementedError

    def build_generator(self, x_data, y_data, batch_size=None):
        """
        Use this method to build a generator.
        :param x_data:
        :param y_data:
        :param batch_size:
        :return:
        """
        batch_size = batch_size or self.config.get('batch_size')
        batches = math.ceil(len(x_data) / batch_size)
        while True:
            for i in range(int(batches)):
                yield x_data[i * batch_size: (i + 1) * batch_size], \
                      y_data[i * batch_size: (i + 1) * batch_size]

    def run(self):
        """
        This methods firstly init model class using flag values, then call model's train method to fit training data.
        :return:
        """
        # prepare data
        data = self.prepare_data()
        # unpack data
        self.train_data, self.eval_data, self.train_size, self.eval_size = None, None, 0, 0
        if not isinstance(data, tuple):
            data = data,
        if len(data) == 1:
            self.train_data = data[0]
        elif len(data) == 2:
            self.train_data, self.eval_data = data
        elif len(data) == 3:
            self.train_data, self.eval_data, self.train_size = data
        elif len(data) == 4:
            self.train_data, self.eval_data, self.train_size, self.eval_size = data

        # build model and run
        model = self.model_class(self.config)
        model.logger = self.logger

        # fit for generator
        if isinstance(self.train_data, types.GeneratorType):
            model.train(self.train_data, self.eval_data, use_generator=True,
                        train_size=self.train_size, eval_size=self.eval_size)
        # fit for normal data
        else:
            model.train(self.train_data, self.eval_data)
