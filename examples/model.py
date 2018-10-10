from model_zoo.model import BaseModel
import tensorflow as tf


class BostonHousingModel(BaseModel):
    def __init__(self, config):
        super(BostonHousingModel, self).__init__(config)
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=None, mask=None):
        o = self.dense(inputs)
        return o
