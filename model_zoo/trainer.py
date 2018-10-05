import tensorflow as tf

tfe = tf.contrib.eager
tf.enable_eager_execution()

tf.flags.DEFINE_integer('batch_size', 32, help='Batch size', allow_override=True)
tf.flags.DEFINE_float('learning_rate', 0.01, help='Learning Rate', allow_override=True)
tf.flags.DEFINE_integer('early_stop_patience', 20, help='Early Stop Patience', allow_override=True)
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints', help='Data source dir', allow_override=True)
tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name', allow_override=True)
tf.flags.DEFINE_bool('checkpoint_restore', True, help='Model restore', allow_override=True)
tf.flags.DEFINE_integer('checkpoint_save_freq', 2, help='Save model every epoch number', allow_override=True)
tf.flags.DEFINE_string('events_dir', 'events', help='TensorBoard events dir', allow_override=True)
tf.flags.DEFINE_integer('epochs', 100, help='Max Epochs', allow_override=True)


class BaseTrainer(object):
    """
    Base Trainer using eager mode, you can override tf.flags to define new config
    and you need to implement prepare_data method to prepare train_data and eval_data
    """
    
    def __init__(self):
        """
        you need to define model_class in your Trainer
        """
        self.model_class = None
        self.flags = tf.flags.FLAGS
    
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
        # init model and train
        if not self.model_class or not self.flags:
            raise Exception('You must define `model_class`')
        model = self.model_class(self.flags.flag_values_dict())
        model.init()
        model.train(self.train_data, self.eval_data)
