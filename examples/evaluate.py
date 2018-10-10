from model_zoo.evaluater import BaseEvaluater
import tensorflow as tf

tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name')


class Evaluater(BaseEvaluater):
    
    def prepare_data(self):
        from tensorflow.python.keras.datasets import boston_housing
        from sklearn.preprocessing import StandardScaler
        (x_train, y_train), (x_eval, y_eval) = boston_housing.load_data()
        ss = StandardScaler()
        ss.fit(x_train)
        x_eval = ss.transform(x_eval)
        return x_eval, y_eval


if __name__ == '__main__':
    result = Evaluater().run()
    print(result)
