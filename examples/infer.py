from model import BostonHousingModel
from model_zoo.inferer import BaseInferer
import tensorflow as tf

tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt-4', help='Model name')


class Inferer(BaseInferer):
    def __init__(self):
        BaseInferer.__init__(self)
        self.model_class = BostonHousingModel
    
    def prepare_data(self):
        from tensorflow.python.keras.datasets import boston_housing
        from sklearn.preprocessing import StandardScaler
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        ss = StandardScaler()
        ss.fit(x_train)
        x_test = ss.transform(x_test)
        return x_test


if __name__ == '__main__':
    result = Inferer().run()
    print(result)
