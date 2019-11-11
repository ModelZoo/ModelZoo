from model_zoo.logger import get_logger
from model_zoo.utils import load_config, load_model, find_model
from model_zoo import flags

flags.DEFINE_string('checkpoint_dir', 'checkpoints', help='Data source dir', allow_override=True)
flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name', allow_override=True)


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
        # get test_data
        self.test_data = self.prepare_data()
        # print()
        # init configs from checkpoints json file and flags
        config = load_config(self.config)
        # init model class
        model_class_name, model_file_name = config.get('model_class_name'), config.get('model_file_name')
        self.model_class = find_model(model_class_name, model_file_name)
        # init model
        model = self.model_class(config)
        # init variables
        model.init()
        # restore model if exists
        load_model(model, self.config.get('checkpoint_dir'), self.config.get('checkpoint_name'))
        # infer
        return model.infer(self.test_data)
