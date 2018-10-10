import tensorflow as tf
from model_zoo.trainer import BaseTrainer
from model_zoo.preprocess import standardize

tf.flags.DEFINE_integer('epochs', 20, 'Max epochs')
tf.flags.DEFINE_string('model_class', 'BostonHousingModel', 'Model class name')


class Trainer(BaseTrainer):
    
    def prepare_data(self):
        from tensorflow.python.keras.datasets import boston_housing
        (x_train, y_train), (x_eval, y_eval) = boston_housing.load_data()
        x_train, x_eval = standardize(x_train, x_eval)
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return train_data, eval_data

if __name__ == '__main__':
    Trainer().run()
