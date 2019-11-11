from model_zoo.model import BaseModel
import tensorflow as tf


class HousePricePredictionModel(BaseModel):
    """
    HousePricePredictionModel
    """
    def __init__(self, config):
        """
        Init model.
        :param config:
        """
        super(HousePricePredictionModel, self).__init__(config)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """
        Build model.
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        o = self.dense(inputs)
        return o
