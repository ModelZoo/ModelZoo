from pathlib import Path

import tensorflow as tf
import json
import numpy as np
from os.path import join, exists, dirname
from os import makedirs
from model_zoo.utils import load_model


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Save model to checkpoints.
    """

    def __init__(self,
                 checkpoint_config,
                 checkpoint_dir,
                 checkpoint_name,
                 checkpoint_restore=True,
                 checkpoint_save_freq=2,
                 monitor='val_loss',
                 verbose=1,
                 checkpoint_save_best=True,
                 checkpoint_save_every=True,
                 checkpoint_save_latest=True,
                 checkpoint_save_weights_only=True,
                 mode='auto',
                 period=1):
        """
        init checkpoint callback
        :param checkpoint_config:
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
        self.checkpoint_config = checkpoint_config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.file_path = join(checkpoint_dir, checkpoint_name)
        self.checkpoint_restore = checkpoint_restore
        self.checkpoint_save_freq = checkpoint_save_freq
        self.checkpoint_save_best = checkpoint_save_best
        self.checkpoint_save_every = checkpoint_save_every
        self.checkpoint_save_latest = checkpoint_save_latest
        self.checkpoint_save_weights_only = checkpoint_save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            self.model.logger.warning('ModelCheckpoint mode %s is unknown, '
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
        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)
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
            file_dir = dirname(self.file_path)
            # ensure file dir exists
            if not exists(file_dir): makedirs(file_dir)
            # get val loss
            val_loss = logs.get(self.monitor)
            path = Path(self.file_path)

            print('path', path.suffix)

            # save best model
            if self.checkpoint_save_best:
                file_stem = '%s-best' % path.stem
                current = val_loss
                if current is None:
                    print('\nCan save best model only with %s available, '
                          'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving best model to %s%s' \
                                  % (epoch + 1, self.monitor, self.best,
                                     current, file_stem, path.suffix))
                        self.best = current
                        # save weights only
                        if self.checkpoint_save_weights_only:
                            self.model.save_weights(join(dirname(self.file_path), '%s%s' % (file_stem, path.suffix)))
                        # save h5 file
                        else:
                            self.model.save(join(dirname(self.file_path), '%s%s' % (file_stem, path.suffix)))
                        # save config align
                        json.dump(dict(self.checkpoint_config, **{'epoch': epoch, 'val_loss': current}),
                                  open(join(dirname(self.file_path), '%s.json' % file_stem), 'w', encoding='utf-8'),
                                  indent=2)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))

            # save latest model
            if self.checkpoint_save_latest:
                file_stem = '%s-latest' % path.stem
                if self.verbose > 0:
                    print('\nEpoch %05d: saving latest model to %s%s' % (epoch + 1, file_stem, path.suffix))
                # save weights only
                if self.checkpoint_save_weights_only:
                    self.model.save_weights(join(dirname(self.file_path), '%s%s' % (file_stem, path.suffix)))
                # save h5 file
                else:
                    self.model.save(join(dirname(self.file_path), '%s%s' % (file_stem, path.suffix)))
                # save config align
                json.dump(dict(self.checkpoint_config, **{'epoch': epoch, 'val_loss': val_loss}),
                          open(join(dirname(self.file_path), '%s.json' % file_stem), 'w', encoding='utf-8'),
                          indent=2)

            # save every model per freq
            if self.checkpoint_save_every:
                if (epoch + 1) % self.checkpoint_save_freq == 0:
                    # new file path, such as `checkpoints/model.ckpt-1`
                    file_stem = '%s-%d' % (path.stem, epoch + 1)
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s%s' % (
                            epoch + 1, file_stem, path.suffix))
                    # save weights only
                    if self.checkpoint_save_weights_only:
                        self.model.save_weights(join(dirname(self.file_path), '%s%s' % (file_stem, path.suffix)))
                    # save h5 file
                    else:
                        self.model.save(join(dirname(self.file_path), '%s%s' % (file_stem, path.suffix)))
                    json.dump(dict(self.checkpoint_config, **{'epoch': epoch, 'val_loss': val_loss}),
                              open(join(dirname(self.file_path), '%s.json' % file_stem), 'w', encoding='utf-8'),
                              indent=2)
