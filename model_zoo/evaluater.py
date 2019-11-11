import tensorflow as tf
from model_zoo.utils import load_config, load_model, find_model
from absl import flags

flags.DEFINE_string('checkpoint_dir', 'checkpoints', help='Data source dir', allow_override=True)
flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name', allow_override=True)


class BaseEvaluater(object):
    """
    Base Evaluater, you need to specify
    """
    
    def __init__(self):
        """
        you need to define model_class in your Inferer
        """
        self.flags = flags.FLAGS
    
    def prepare_data(self):
        """
        you need to implement this method
        :return:
        """
        raise NotImplementedError
    
    def run(self):
        """
        start inferring
        :return:
        """
        # prepare data
        self.eval_data = self.prepare_data()
        # split data
        x_eval, y_eval = self.eval_data
        # init configs from checkpoints json file and flags
        config = load_config(self.flags)
        # init model class
        model_class_name, model_file = config['model_class'], config['model_file']
        self.model_class = find_model(model_class_name, model_file)
        # init model
        model = self.model_class(config)
        # init variables
        model.construct(x_eval)
        # restore model
        load_model(model, self.flags.checkpoint_dir, self.flags.checkpoint_name)
        # evaluate
        return model.evaluate(x_eval, y_eval)
