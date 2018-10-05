import tensorflow as tf

tfe = tf.contrib.eager
import model_zoo.callbacks as callbacks


class BaseModel(tf.keras.Model):
    """
    Base Keras Model, you can inherit this Model and
    override '__init__()', 'call()', 'init()', 'callback()' methods
    """
    def __init__(self, config):
        """
        init config, batch_size, epochs
        :param config:
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
    
    def call(self, inputs, training=None, mask=None):
        """
        build your models
        :param inputs: inputs x
        :param training: training flag
        :param mask: mask flag
        :return: y_pred
        """
        return inputs
    
    def init(self):
        """
        default call compile method using sgd optimizer and mse loss
        :return:
        """
        self.compile(optimizer=self.optimizer(), loss='mse')
    
    def train(self, train_data, eval_data=None):
        """
        train and fit model
        :param train_data: x, y data pairs for training
        :param eval_data: x, y data pairs for evaluating
        :return: fit result
        """
        x, y = train_data
        return self.fit(x=x,
                        y=y,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_data=eval_data,
                        callbacks=self.callbacks())
    
    def infer(self, test_data, batch_size=None):
        """
        do inference, default call predict method
        :param test_data: x data
        :param batch_size: batch_size
        :return:
        """
        x = test_data
        if not batch_size:
            batch_size = self.batch_size
        return self.predict(x, batch_size)
    
    def callbacks(self):
        """
        default callbacks, including logger, early stop, tensor board, checkpoint
        :return:
        """
        cbs = []
        cbs.append(tf.keras.callbacks.BaseLogger())
        cbs.append(tf.keras.callbacks.EarlyStopping(patience=self.config.get('early_stop_patience', 50)))
        cbs.append(tf.keras.callbacks.TensorBoard(log_dir=self.config.get('events_dir', 'events')))
        cbs.append(callbacks.ModelCheckpoint(
            checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
            checkpoint_name=self.config.get('checkpoint_name', 'model'),
            checkpoint_restore=self.config.get('checkpoint_restore', True),
            checkpoint_save_freq=self.config.get('checkpoint_save_freq', 2),
            checkpoint_save_best_only=self.config.get('checkpoint_save_best_only', False)))
        return cbs
    
    def optimizer(self):
        """
        build optimizer, default to sgd and lr 0.01
        :return:
        """
        return tf.train.GradientDescentOptimizer(self.config.get('learning_rate', 0.01))
