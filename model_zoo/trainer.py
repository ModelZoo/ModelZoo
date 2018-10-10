import tensorflow as tf
from model_zoo.utils import find_model

tfe = tf.contrib.eager
tf.enable_eager_execution()

# ========== Base Configs =================
tf.flags.DEFINE_integer('batch_size', 32, help='Batch size', allow_override=True)
tf.flags.DEFINE_float('learning_rate', 0.01, help='Learning rate', allow_override=True)
tf.flags.DEFINE_integer('epochs', 100, help='Max epochs', allow_override=True)

# ========== Early Stop Configs ================
tf.flags.DEFINE_bool('early_stop_enable', True, help='Whether to enable early stop', allow_override=True)
tf.flags.DEFINE_integer('early_stop_patience', 20, help='Early stop patience', allow_override=True)

# ========== Checkpoint Configs =================
tf.flags.DEFINE_bool('checkpoint_enable', True, help='Whether to save model checkpoint', allow_override=True)
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints', help='Data source dir', allow_override=True)
tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name', allow_override=True)
tf.flags.DEFINE_bool('checkpoint_restore', False, help='Model restore', allow_override=True)
tf.flags.DEFINE_integer('checkpoint_save_freq', 2, help='Save model every epoch number', allow_override=True)

# ========== TensorBoard Events Configs =================
tf.flags.DEFINE_bool('tensor_board_enable', True, help='Whether to enable TensorBoard events', allow_override=True)
tf.flags.DEFINE_string('tensor_board_dir', 'events', help='TensorBoard events dir', allow_override=True)

# ========== Other Basic Configs ==================
tf.flags.DEFINE_string('model_file', 'model', help='path of model file which including model class',
                       allow_override=True)


class BaseTrainer(object):
    """
    Base Trainer using eager mode, you can override tf.flags to define new config
    and you need to implement prepare_data method to prepare train_data and eval_data
    """
    
    def __init__(self):
        """
        you need to define model_class in your Trainer
        """
        self.flags = tf.flags.FLAGS
        # init model class
        model_class_name = self.flags.model_class
        self.model_class = find_model(model_class_name, self.flags.model_file)
    
    def prepare_data(self):
        """
        you need to implement this method
        :return:
        """
        raise NotImplementedError
    
    def run(self):
        """
        this methods firstly init model class using flag values, then call model's train method to fit training data
        :return:
        """
        # prepare data
        self.train_data, self.eval_data = self.prepare_data()
        model = self.model_class(self.flags.flag_values_dict())
        model.init()
        model.train(self.train_data, self.eval_data)
