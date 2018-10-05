import logging
import tensorflow as tf
import json
import numpy as np
from os.path import join
from model_zoo.utils import load_model

tfe = tf.contrib.eager


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Save model to checkpoints
    """
    
    def __init__(self,
                 checkpoint_dir,
                 checkpoint_name,
                 checkpoint_restore=True,
                 checkpoint_save_freq=2,
                 monitor='val_loss',
                 verbose=1,
                 checkpoint_save_best_only=False,
                 mode='auto',
                 period=1):
        """
        init checkpoint callback
        :param checkpoint_dir:
        :param checkpoint_name:
        :param checkpoint_restore:
        :param checkpoint_save_freq:
        :param monitor:
        :param verbose:
        :param checkpoint_save_best_only:
        :param mode:
        :param period:
        """
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.file_path = join(checkpoint_dir, checkpoint_name)
        self.checkpoint_restore = checkpoint_restore
        self.checkpoint_save_freq = checkpoint_save_freq
        self.checkpoint_save_best_only = checkpoint_save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        
        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    
    def on_train_begin(self, logs=None):
        """
        restore model from checkpoints
        :param logs:
        :return:
        """
        if self.checkpoint_restore:
            load_model(self.model, self.checkpoint_dir, self.checkpoint_name)
    
    def on_epoch_end(self, epoch, logs=None):
        """
        save model on epoch end
        :param epoch:
        :param logs:
        :return:
        """
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            file_path = self.file_path.format(epoch=epoch + 1, **logs)
            if self.checkpoint_save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('\nCan save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, file_path))
                        self.best = current
                        tfe.Saver(self.model.variables).save(file_path)
                        json.dump(dict(self.model.config),
                                  open('%s.json' % file_path, 'w', encoding='utf-8'),
                                  indent=2)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, file_path))
                tfe.Saver(self.model.variables).save(file_path)
                json.dump(dict(self.model.config),
                          open('%s.json' % file_path, 'w', encoding='utf-8'),
                          indent=2)
                if (epoch + 1) % self.checkpoint_save_freq == 0:
                    if self.verbose > 0:
                        print('Epoch %05d: saving model to %s-%d' % (epoch + 1, file_path, epoch + 1))
                    tfe.Saver(self.model.variables).save(file_path, global_step=epoch + 1)
                    json.dump(dict(self.model.config),
                              open('%s-%d.json' % (file_path, epoch + 1), 'w', encoding='utf-8'),
                              indent=2)
