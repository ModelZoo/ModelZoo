from model_zoo.logger import get_logger
from model_zoo.utils import load_config, load_model, find_model_class
from absl import flags

# ========== Checkpoint ================
flags.DEFINE_string('checkpoint_dir', 'checkpoints', help='Data source dir', allow_override=True)
flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name', allow_override=True)

# ========== Log System ================
flags.DEFINE_bool('log_enable', True, help='Whether to enable Log System', allow_override=True)
flags.DEFINE_string('log_level', 'DEBUG', help='Log Level', allow_override=True)
flags.DEFINE_string('log_rotation', '100MB', help='Log file rotation', allow_override=True)
flags.DEFINE_string('log_retention', None, help='Log file retention', allow_override=True)
flags.DEFINE_string('log_format', '{time} - {level} - {module} - {file} - {message}', help='Log record format',
                    allow_override=True)
flags.DEFINE_string('log_folder', './logs/', help='Folder of log file', allow_override=True)
flags.DEFINE_string('log_file', 'evaluate.log', help='Name of log file', allow_override=True)
flags.DEFINE_string('log_path', '', help='File path of log file', allow_override=True)


class BaseEvaluater(object):
    """
    Base Evaluater, you need to specify
    """
    
    def __init__(self):
        """
        you need to define model_class in your Inferer
        """
        self.config = flags.FLAGS.flag_values_dict()
        
        # get logger
        logger = get_logger(self.config)
        self.logger = logger
    
    def data(self):
        """
        you need to implement this method
        :return:
        """
        raise NotImplementedError
    
    def run(self, **kwargs):
        """
        start inferring
        :return:
        """
        # prepare data
        self.eval_data = self.data()
        # split data
        x_eval, y_eval = self.eval_data
        # init configs from checkpoints json file and flags
        config = load_config(self.config)
        # init model class
        model_class_name, model_file_name = config.get('model_class_name'), config.get('model_file_name')
        self.model_class = find_model_class(model_class_name, model_file_name)
        
        # init model
        model = self.model_class(config=config)
        model.logger = self.logger
        self.logger.info(f'initialize model logger {model.logger} of {model}')
        
        # restore model
        load_model(model, self.config.get('checkpoint_dir'), self.config.get('checkpoint_name'))
        # evaluate
        return model.evaluate(x_eval, y_eval, **kwargs)
