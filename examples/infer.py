from model_zoo.inferer import BaseInferer
from model_zoo.preprocess import standardize
import tensorflow as tf

tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt-20', help='Model name')


class Inferer(BaseInferer):
    
    def prepare_data(self):
        from tensorflow.python.keras.datasets import boston_housing
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        _, x_test = standardize(x_train, x_test)
        return x_test


if __name__ == '__main__':
    result = Inferer().run()
    print(result)
