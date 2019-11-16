from model_zoo import flags, datasets, preprocess
from model_zoo.trainer import BaseTrainer

flags.define('epochs', 100)
flags.define('model_class_name', 'HousePricePredictionModel')
flags.define('checkpoint_name', 'model.ckpt')


class Trainer(BaseTrainer):
    
    def data(self):
        (x_train, y_train), (x_eval, y_eval) = datasets.boston_housing.load_data()
        x_train, x_eval = preprocess.standardize(x_train, x_eval)
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return train_data, eval_data


if __name__ == '__main__':
    Trainer().run()
