from model_zoo.evaluater import BaseEvaluater
from model_zoo import flags, preprocess, datasets

flags.define('checkpoint_name', 'model-20.ckpt', help='Model name')


class Evaluater(BaseEvaluater):
    
    def data(self):
        (x_train, y_train), (x_eval, y_eval) = datasets.boston_housing.load_data()
        x_train, x_eval = preprocess.standardize(x_train, x_eval)
        return (x_eval, y_eval)

if __name__ == '__main__':
    result = Evaluater().run()
    print(result)
