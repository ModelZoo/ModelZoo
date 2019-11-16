# PricePrediction

Simple Linear Regression Model implemented by [ModelZoo](https://github.com/ModelZoo/ModelZoo).

## Installation

Firstly you need to clone this repository and install dependencies with pip:

```
pip3 install -r requirements.txt
```

## Dataset

We use BostonHousing dataset for example.

## Usage

We can run this model like this:

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

It runs only 42 epochs and stopped early, because there are no more good evaluation results for 20 epochs.

When finished, we can find two folders generated named `checkpoints` and `events`.

Go to `events` and run TensorBoard:

```
cd events
tensorboard --logdir=.
```

TensorBoard like this:

![](https://ws4.sinaimg.cn/large/006tNbRwgy1fvxrcajse2j31kw0hkgnf.jpg)

There are training batch loss, epoch loss, eval loss.

And also we can find checkpoints in `checkpoints` dir.

It saved the best model named `model.ckpt` according to eval score, and it also saved checkpoints every 2 epochs.

Next we can predict using existing checkpoints and `infer.py`.

Now we've restored the specified model `model-best.ckpt` and prepared test data, outputs like this:

```python
[[ 9.637125 ]
 [21.368305 ]
 [20.898445 ]
 [33.832504 ]
 [25.756516 ]
 [21.264557 ]
 [29.069794 ]
 [24.968184 ]
 ...
 [36.027283 ]
 [39.06852  ]
 [25.728745 ]
 [41.62165  ]
 [34.340042 ]
 [24.821484 ]]
```

OK, we've finished restoring and predicting. Just so quickly.

## License

MIT
