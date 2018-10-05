# ModelZoo

A framework to help you build model much more easily.

## Installation

You can install this package easily with pip:

```
pip3 install model-zoo
```

## Usage

Let's implement a linear-regression model quickly.

Here we use boston_housing dataset as example.

Define a linear model like this, named `model.py`:

```
from model_zoo.model import BaseModel
import tensorflow as tf

class BostonHousingModel(BaseModel):
    def __init__(self, config):
        super(BostonHousingModel, self).__init__(config)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        o = self.dense(inputs)
        return o

```

Then define a trainer like this, named `train.py`:

```


```

Now, we've finished this model.

Next we can run this model like this:

```
python3 train.py
```

Outputs like this:

```

Epoch 1/100
 1/13 [=>............................] - ETA: 0s - loss: 816.1798
13/13 [==============================] - 0s 4ms/step - loss: 457.9925 - val_loss: 343.2489

Epoch 2/100
 1/13 [=>............................] - ETA: 0s - loss: 361.5632
13/13 [==============================] - 0s 3ms/step - loss: 274.7090 - val_loss: 206.7015
Epoch 00002: saving model to checkpoints/model.ckpt

Epoch 3/100
 1/13 [=>............................] - ETA: 0s - loss: 163.5308
13/13 [==============================] - 0s 3ms/step - loss: 172.4033 - val_loss: 128.0830

Epoch 4/100
 1/13 [=>............................] - ETA: 0s - loss: 115.4743
13/13 [==============================] - 0s 3ms/step - loss: 112.6434 - val_loss: 85.0848
Epoch 00004: saving model to checkpoints/model.ckpt

Epoch 5/100
 1/13 [=>............................] - ETA: 0s - loss: 149.8252
13/13 [==============================] - 0s 3ms/step - loss: 77.0281 - val_loss: 57.9716
....

Epoch 42/100
 7/13 [===============>..............] - ETA: 0s - loss: 20.5911
13/13 [==============================] - 0s 8ms/step - loss: 22.4666 - val_loss: 23.7161
Epoch 00042: saving model to checkpoints/model.ckpt
```

