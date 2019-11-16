from model_zoo import flags, datasets, preprocess
from model_zoo.inferer import BaseInferer

flags.define('checkpoint_name', 'model-best.ckpt')


class Inferer(BaseInferer):
    
    def data(self):
        (x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data()
        _, x_test = preprocess.standardize(x_train, x_test)
        return x_test


if __name__ == '__main__':
    result = Inferer().run()
    print(result)
