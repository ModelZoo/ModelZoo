import tensorflow as tf
from model_zoo.trainer import BaseTrainer
from model_zoo.preprocess import standardize
from model_zoo import flags

flags.DEFINE_integer('epochs', 20, 'Max epochs')
flags.DEFINE_string('model_file_name', 'model', help='Path of model file which including model class')
flags.DEFINE_string('model_class_name', 'HousePricePredictionModel', 'Model class name')
flags.DEFINE_string('logger_level', 'INFO', 'Log level')


class Trainer(BaseTrainer):
    """
    Train Price Prediction Model.
    """

    def prepare_data(self):
        """
        Prepare train data.
        :return:
        """
        (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.boston_housing.load_data()
        x_train, x_eval = standardize(x_train, x_eval)
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        self.logger.debug('x_train data shape %s, y_train %s' % (x_train.shape, y_train.shape))
        return train_data, eval_data


if __name__ == '__main__':
    Trainer().run()
