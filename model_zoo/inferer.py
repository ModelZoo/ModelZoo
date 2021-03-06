from model_zoo.logger import get_logger
from model_zoo.utils import load_config, load_model, find_model_class
from model_zoo import flags

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
flags.DEFINE_string('log_file', 'infer.log', help='Name of log file', allow_override=True)
flags.DEFINE_string('log_path', '', help='File path of log file', allow_override=True)


class BaseInferer():
    """
    Base Inferer, you need to specify
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
        # get test_data
        self.test_data = self.data()
        # init configs from checkpoints json file and flags
        config = load_config(self.config)
        # init model class
        model_class_name, model_file_name = config.get('model_class_name'), config.get('model_file_name')
        self.model_class = find_model_class(model_class_name, model_file_name)
        # init model
        model = self.model_class(config=config)
        # restore model if exists
        load_model(model, self.config.get('checkpoint_dir'), self.config.get('checkpoint_name'))
        # infer
        return model.infer(self.test_data, **kwargs)
