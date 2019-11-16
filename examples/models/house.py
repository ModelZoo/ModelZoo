from model_zoo import Model
import tensorflow as tf

class HousePricePredictionModel(Model):
    
    def inputs(self):
        return tf.keras.Input(shape=(13))
    
    def outputs(self, inputs):
        return tf.keras.layers.Dense(1)(inputs)
