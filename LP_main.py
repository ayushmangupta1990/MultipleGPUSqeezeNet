#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:23:13 2018

@author: vishalp
"""
import os
import itertools, six, argparse
import tensorflow as tf
from LP_model import squeeze_net
import LP_device_util
from LP_dataset_util import input_batch_fn, input_batch_fn_folder, input_batch_fn_csv
tf.logging.set_verbosity(tf.logging.INFO)

def get_model_fn(num_gpus, variable_strategy, num_workers):
  """Returns a function that will build the squeezenet model."""

  def _squeeze_net_gpu(features, labels, mode, params):
    """Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    #weight_decay = params.weight_decay
    #momentum = params.momentum

    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_gradvars = []
    tower_preds = []
    tower_probs = []

    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = LP_device_util.local_device_setter(
            worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = LP_device_util.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
      with tf.variable_scope('squeezenet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss, gradvars, preds, probs = _tower_fn(
                is_training, tower_features[i], tower_labels[i], params)
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            tower_preds.append(preds)
            tower_probs.append(probs)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from
              # the 1st tower. Ideally, we should grab the updates from all
              # towers but these stats accumulate extremely fast so we can
              # ignore the other stats from the other towers without
              # significant detriment.
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             name_scope)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
      all_grads = {}
      for grad, var in itertools.chain(*tower_gradvars):
        if grad is not None:
          all_grads.setdefault(var, []).append(grad)
      for var, grads in six.iteritems(all_grads):
        # Average gradients on the same device as the variables
        # to which they apply.
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
        gradvars.append((avg_grad, var))

    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):
      # Suggested learning rate scheduling from
      # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
      loss = tf.reduce_mean(tower_losses, name='loss')
      optimizer=tf.train.AdamOptimizer(learning_rate=params.learning_rate)
      train_hooks = []
      if params.sync:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer, replicas_to_aggregate=num_workers)
        sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
        train_hooks.append(sync_replicas_hook)

      # Create single grouped train op
      train_op = [optimizer.apply_gradients(gradvars, global_step=tf.train.get_global_step())]
      train_op.extend(update_ops)
      train_op = tf.group(*train_op)

      predictions = tf.concat([p for p in tower_preds], axis=0)
      stacked_labels = tf.concat([l for l in labels], axis=0)
      stacked_labels = tf.argmax(stacked_labels, axis = 1)
      metrics = {
          'accuracy':
              tf.metrics.accuracy(stacked_labels, predictions)
      }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)

  return _squeeze_net_gpu


def _tower_fn(is_training, feature, label, params):
    reuse=tf.AUTO_REUSE
    logits = squeeze_net(feature, is_training,reuse)
    
    tower_probs = tf.nn.softmax(logits, name = 'softmax_layer')
    tower_preds = tf.argmax(logits, axis = 1)
    tower_loss = tf.losses.softmax_cross_entropy(label, logits)
    tower_loss = tf.reduce_mean(tower_loss)
    model_params = tf.trainable_variables()
    tower_grads = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grads, model_params), tower_preds, tower_probs


def get_experiment_fn(num_gpus, variable_strategy):
  """Returns an Experiment function.

  Experiments perform training on several workers in parallel,
  in other words experiments know how to invoke train and eval in a sensible
  fashion for distributed training. Arguments passed directly to this
  function are not tunable, all other arguments should be passed within
  tf.HParams, passed to the enclosed function.

  Args:
      data_dir: str. Location of the data for input_fns.
      num_gpus: int. Number of GPUs on each worker.
      variable_strategy: String. CPU to use CPU as the parameter server
      and GPU to use the GPUs as the parameter server.
      use_distortion_for_training: bool. See cifar10.Cifar10DataSet.
  Returns:
      A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
      tf.contrib.learn.Experiment.

      Suitable for use by tf.contrib.learn.learn_runner, which will run various
      methods on Experiment (train, evaluate) based on information
      about the current runner in `run_config`.
  """

  def _experiment_fn(run_config, hparams):
    """Returns an Experiment."""
    # Create estimator.
    #train_input_fn = input_batch_fn(hparams.train_npy,hparams.num_epochs, hparams.train_batch_size, num_gpus)
    #eval_input_fn = input_batch_fn(hparams.eval_npy,hparams.num_epochs,hparams.eval_batch_size, num_gpus)
    # train_input_fn = input_batch_fn_folder(hparams.train_data_dir,hparams.num_epochs, hparams.train_batch_size, num_gpus)
    # eval_input_fn = input_batch_fn_folder(hparams.eval_data_dir,hparams.num_epochs,hparams.eval_batch_size, num_gpus)    
    train_input_fn = input_batch_fn_csv(hparams.train_csv,hparams.num_epochs, hparams.train_batch_size, num_gpus)
    eval_input_fn = input_batch_fn_csv(hparams.eval_csv,hparams.num_epochs, hparams.train_batch_size, num_gpus)
    train_steps = hparams.train_steps
    eval_steps = hparams.eval_steps
 
    classifier = tf.estimator.Estimator(
        model_fn=get_model_fn(num_gpus, variable_strategy,
                              run_config.num_worker_replicas or 1),
        config=run_config,
        params=hparams)

    # Create experiment.
    return tf.contrib.learn.Experiment(
        classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        min_eval_frequency = 400,
        train_steps=train_steps,
        eval_steps=eval_steps)

  return _experiment_fn


def main(model_dir,num_gpus, variable_strategy,log_device_placement, num_intra_threads,**hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = LP_device_util.RunConfig(
      session_config=sess_config, model_dir=model_dir,log_step_count_steps=10,save_checkpoints_steps = 240)
  tf.contrib.learn.learn_runner.run(
      get_experiment_fn(num_gpus, variable_strategy),
      run_config=config,
      hparams=tf.contrib.training.HParams(
          is_chief=config.is_chief,
          **hparams))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
  parser.add_argument('--job-dir', type = str, required = True, help='directory to package trainer')
  parser.add_argument(
      '--train_data_dir',
      type=str,
      help='The base directory for training file.')
  parser.add_argument(
      '--eval_data_dir',
      type=str,
      help='The base directory for eval files.')
  parser.add_argument('--train_npy', type = str, help = 'files when reading file from npy')
  parser.add_argument('--eval_npy', type = str, help = 'files when reading file from npy')
  parser.add_argument('--train_csv', type = str, help = 'files when reading file from csv')
  parser.add_argument('--eval_csv', type = str, help = 'files when reading file from csv')  
  parser.add_argument(
      '--variable_strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='CPU',
      help='Where to locate variable operations')
  parser.add_argument(
      '--num_gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument('--num_epochs', type = int, help=' use either epochs or steps')
  parser.add_argument(
      '--train_steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
  parser.add_argument('--eval_steps', type = int, default = 60, help = 'no of eval batches to run')
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=64,
      help='Batch size for training.')
  parser.add_argument(
      '--eval_batch_size',
      type=int,
      default=100,
      help='Batch size for validation.')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
  parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present when running in a distributed environment will run on sync mode.\
      """)
  parser.add_argument(
      '--num_intra_threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
  parser.add_argument(
      '--num_inter_threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
  parser.add_argument(
      '--log_device_placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')

  args = parser.parse_args()

  
  main(**vars(args))
  
