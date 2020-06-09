# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#@title Imports, configurations, and helper functions { display-mode: "form" }
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import itertools
import os
import pickle

import argparse

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import tensorflow as tf
from tensorflow.python.ops.parallel_for import gradients
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import sklearn.linear_model as sk_linear

parser = argparse.ArgumentParser(description='Representation Learning Experiments')
parser.add_argument('--dataset', type=str, 
                   help='cifar10 or mnist')
parser.add_argument('--lr', type=float, 
                   help='1e-4 or 1e-5')
parser.add_argument('--batch_size', type=int, 
                   help='training batch size')

args = parser.parse_args()

# learning rate for ours CIFAR10 is 1e-4, otherwise follows below

TFDS_NAME = args.dataset #"cifar10" # mnist or cifar10
NRUNS = 10 #@param { type: "slider", min: 1, max: 20, step: 1}

# parameters for training
if TFDS_NAME == "mnist":
  DIMS = 784
elif TFDS_NAME == "cifar10":
  DIMS = 3072
LEARNING_RATE = args.lr #1e-5
N_CLASSES = 10
TRAIN_BATCH_SIZE = args.batch_size #64 #@param { type: "slider", min: 64, max: 128, step: 64}

# save results
RESULT_DIR = '{}_nrun={}_lr={}_batch={}.pkl'.format(TFDS_NAME, NRUNS, LEARNING_RATE, TRAIN_BATCH_SIZE)

# Do not change these
RUN_EXPERIMENTS = True #@param { type: "boolean"}
FEATURE_INPUT = "image"
FEATURE_LABEL = "label"

#slim = tf.contrib.slim
tfb = tfp.bijectors
tfd = tfp.distributions
tfkl = tf.keras.layers

tf.keras.backend.clear_session()

ResultsConfig = collections.namedtuple(
    "ResultsConfig", ["nets", "critic", "loss"])

Results = collections.namedtuple(
    'Results',
    ['iterations', 'training_losses', 'testing_losses',
     'classification_accuracies', 'singular_values'])

ResultsAdversarial = collections.namedtuple(
    "ResultsAdversarial",
    ["losses_e", "losses_c", "classification_accuracies", "iters"]
)

ResultsSamplingIssues = collections.namedtuple(
    "ResultsSamplingIssues", ["mi_true", "nce_estimates_noniid", 
                              "nce_estimates_iid", "nwj_estimates_noniid", 
                              "nwj_estimates_iid"])

def convert_to_data_frame(result, exp_name, nets, critic, loss, seed):
  """Convert results class to a data frame."""
  label = "{}, {}, {}".format(nets, critic, loss)
  rows = list(
      zip(
          itertools.repeat(exp_name),
          itertools.repeat(nets),
          itertools.repeat(critic),
          itertools.repeat(loss),
          itertools.repeat(seed),
          result.iterations,
          [-loss for loss in result.testing_losses],  # Loss -> bound.
          result.classification_accuracies,
          itertools.repeat(label)))
  df_eval = pd.DataFrame(
      rows,
      columns=("exp_name", "nets", "Critic", "Estimator",
               "run", "iteration", "bound_value", "accuracy", "label"))

  df_eval["Estimator"] = df_eval["Estimator"].replace(
      to_replace={
          "cpc": "$CPC$",
          "pcc": "$PCC$",
          "drfc": "$D-RFC$",
          "wpc": "$WPC$"
      })
  df_eval["Critic"] = df_eval["Critic"].replace(
      to_replace={
          "concat": "MLP",
          "separable": "Separable",
          "innerprod": "Inner product",
          "bilinear": "Bilinear"
      })
  return df_eval


def apply_default_style(ax):
  ax.set_xlim([0, 20001])
  ax.get_xaxis().set_major_formatter(
      FuncFormatter(lambda x, p: format(int(x/1000), ',')))
  ax.set_xlabel("Training steps (in thousands)")
  plt.tick_params(top=False, right=False, bottom=False, left=False)
  handles, labels = ax.get_legend_handles_labels()
  plt.legend(loc="lower right", handles=handles[1:], labels=labels[1:])

FONTSIZE = 15 
sns.set_style("whitegrid")
plt.rcParams.update({'axes.labelsize': FONTSIZE,
                     'xtick.labelsize': FONTSIZE,
                     'ytick.labelsize': FONTSIZE,
                     'legend.fontsize': FONTSIZE})


def get_testing_loss(x_array, session, loss, data_ph, dims, batch_size=512):
  total_loss = 0
  for i in range(0, x_array.shape[0], batch_size):
    x_slice = x_array[i:i+batch_size, :dims]
    total_loss += x_slice.shape[0] * session.run(loss,
                                                 feed_dict={data_ph: x_slice})
  return total_loss / x_array.shape[0]

def get_classification_accuracy(session, codes, data_ph, dims):
  x_train_mapped = map_data(x_train, session, codes, data_ph, dims)
  x_test_mapped = map_data(x_test, session, codes, data_ph, dims)
  accuracy = logistic_fit(x_train_mapped, y_train, x_test_mapped, y_test)
  return accuracy

def map_data(x_array, session, codes, data_ph, dims, batch_size=512):
  x_mapped = []
  for i in range(0, x_array.shape[0], batch_size):
    x_mapped.append(
        session.run(codes,
                    feed_dict={data_ph: x_array[i:i+batch_size, :dims]}))
  return np.concatenate(x_mapped, axis=0)


# @title Import bounds implemented by Poole et al. (2019) { display-mode: "form" }
# From https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb 

def reduce_logmeanexp_nodiag(x, axis=None):
  batch_size = x.shape[0]
  logsumexp = tf.reduce_logsumexp(input_tensor=x - tf.linalg.tensor_diag(np.inf * tf.ones(batch_size)), axis=axis)
  if axis:
    num_elem = batch_size - 1.
  else:
    num_elem  = batch_size * (batch_size - 1.)
  return logsumexp - tf.math.log(num_elem)

def tuba_lower_bound(scores, log_baseline=None):
  if log_baseline is not None:
    scores -= log_baseline[:, None]
  batch_size = tf.cast(scores.shape[0], tf.float32)
  # First term is an expectation over samples from the joint,
  # which are the diagonal elmements of the scores matrix.
  joint_term = tf.reduce_mean(input_tensor=tf.linalg.diag_part(scores))
  # Second term is an expectation over samples from the marginal,
  # which are the off-diagonal elements of the scores matrix.
  marg_term = tf.exp(reduce_logmeanexp_nodiag(scores))
  return 1. + joint_term -  marg_term

def nwj_lower_bound(scores):
  # equivalent to: tuba_lower_bound(scores, log_baseline=1.)
  return tuba_lower_bound(scores - 1.) 

@tf.function
def js_fgan_lower_bound(f):
  """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
  f_diag = tf.linalg.tensor_diag_part(f)
  first_term = tf.reduce_mean(-tf.nn.softplus(-f_diag))
  n = tf.cast(f.shape[0], tf.float32)
  second_term = (tf.reduce_sum(tf.nn.softplus(f)) - tf.reduce_sum(tf.nn.softplus(f_diag))) / (n * (n - 1.))
  return first_term - second_term

@tf.function
def infonce_lower_bound(scores):
  """InfoNCE lower bound from van den Oord et al. (2018)."""
  nll = tf.reduce_mean(input_tensor=tf.linalg.diag_part(scores) - tf.reduce_logsumexp(input_tensor=scores, axis=1))
  # Alternative implementation:
  # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
  # tf.print(nll) # mostly around -0.4~-0.6
  mi = tf.math.log(tf.cast(scores.shape[0], tf.float32)) + nll
  return mi

@tf.function
def our_lower_bound(scores):
  # scores: 128, 128
  """Our lower bound"""
  batch_size = tf.cast(scores.shape[0], tf.float32)
  joint_term = tf.reduce_mean(input_tensor=tf.linalg.diag_part(scores))
  
  # expectation
  scores_sq = scores**2
  marg_num = batch_size * (batch_size - 1.)
  marg_term = tf.reduce_sum(input_tensor=scores_sq) - tf.reduce_sum(input_tensor=tf.linalg.diag_part(scores_sq))
  marg_term = marg_term / marg_num
  # tf.print(joint_term - 0.5*marg_term)
  return joint_term - 0.5*marg_term


  # nll = tf.reduce_mean(tf.linalg.diag_part(scores) - 0.5 * tf.math.reduce_euclidean_norm(scores, axis=1))
  # tf.print(nll)
  # mi = tf.math.log(tf.cast(scores.shape[0].value, tf.float32)) + nll
  # return mi


# @title Define the linear evaluation protocol { display-mode: "form" }

def logistic_fit(x_train, y_train, x_test, y_test):
  logistic_regressor = sk_linear.LogisticRegression(
      solver='saga', multi_class='multinomial', tol=.1, C=10.)
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)
  logistic_regressor.fit(x_train, y_train.ravel())
  return logistic_regressor.score(x_test, y_test.ravel())

# @title Define and load the dataset, check baseline in pixel space { display-mode: "form" }

tf.compat.v1.reset_default_graph()


def map_fn(example):
  image = example[FEATURE_INPUT]
  image = tf.cast(image, tf.float32) / 255.0
  image = tf.reshape(image, [-1])  # Flatten.
  label = example[FEATURE_LABEL]
  return {FEATURE_INPUT: image, FEATURE_LABEL: label}

def load_data(split):
  return (tfds.load(TFDS_NAME, split=split)
              .cache()
              .map(map_func=map_fn)
              .shuffle(1000))

def tfds_to_np(dataset):
  features = list(tfds.as_numpy(dataset))
  images = np.stack([f[FEATURE_INPUT].ravel() for f in features])
  labels = np.stack([f[FEATURE_LABEL].ravel() for f in features])
  return images, labels

dataset_train = load_data("train")
dataset_test = load_data("test")
x_train, y_train = tfds_to_np(dataset_train)
x_test, y_test = tfds_to_np(dataset_test)
tf.compat.v1.reset_default_graph()

x_train_noisy = x_train + 0.05 * np.random.randn(*x_train.shape)
x_test_noisy = x_test + 0.05 * np.random.randn(*x_test.shape)
print("Fit on half the pixels: {}. It should be around 0.835.".format(
    logistic_fit(x_train_noisy[:, :DIMS//2], y_train,
                 x_test_noisy[:, :DIMS//2], y_test)))

def processed_train_data(dims, batch_size):
  dataset = load_data("train")
  dataset_batched = dataset.repeat().batch(batch_size, drop_remainder=True)
  get_next = tf.compat.v1.data.make_one_shot_iterator(dataset_batched).get_next()
  features = get_next[FEATURE_INPUT]
  labels = get_next[FEATURE_LABEL]
  # Martin: where the problem occurs
  x_1, x_2 = tf.split(features, [dims, DIMS-dims], axis=-1)
  return x_1, x_2, labels

class MLP(tf.keras.Model):
  def __init__(self, layer_dimensions, shortcuts, dense_kwargs={}):
      super(MLP, self).__init__()
      self._layers = [tfkl.Dense(dimensions, **dense_kwargs)
                     for dimensions in layer_dimensions[:-1]]
      dense_kwargs_copy = copy.deepcopy(dense_kwargs)
      dense_kwargs_copy["activation"] = None
      self._layers.append(tfkl.Dense(layer_dimensions[-1], **dense_kwargs_copy))
      self._shortcuts = shortcuts

  @property
  def layers(self):
    return self._layers

  def __call__(self, inputs):
    x = inputs
    for layer in self.layers:
      x = layer(x) + x if self._shortcuts else layer(x)
    return x


# LayerNorm implementation copied from
# https://stackoverflow.com/questions/39095252/fail-to-implement-layer-normalization-with-keras
class LayerNorm(tfkl.Layer):

    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """
    def __init__(self, scale_initializer='ones', bias_initializer='zeros',
                 axes=[1,2,3], epsilon=1e-6, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.scale_initializer = tf.keras.initializers.get(scale_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.axes = axes

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],),
                                     initializer=self.scale_initializer,
                                     trainable=True,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.bias_initializer,
                                    trainable=True,
                                    name='{}_bias'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        mean = tf.keras.backend.mean(x, axis=self.axes, keepdims=True)
        std = tf.keras.backend.std(x, axis=self.axes, keepdims=True)
        norm = (x - mean) * (1/(std + self.epsilon))
        return norm * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class ConvNet(tf.keras.Sequential):
  def __init__(self, channels=64, kernel_size=5, input_dim=DIMS//2, output_dim=100,
               activation=tf.nn.relu):
      # Note: This works only for the specific data set considered here.
      super(ConvNet, self).__init__([
        tfkl.Reshape((14, 28, 1), input_shape=(input_dim,)),
        tfkl.Conv2D(channels, kernel_size, strides=2,
                    padding="same", activation=activation),
        tfkl.Conv2D(2*channels, kernel_size, strides=2,
                    padding="same", activation=activation),
        LayerNorm(),
        tfkl.GlobalAveragePooling2D(),
        tfkl.Dense(output_dim),
      ])

from tensorflow_probability.python.internal import tensorshape_util
import tensorflow.compat.v1 as tf1
from tensorflow_probability.python.bijectors import affine_scalar
from tensorflow_probability.python.bijectors import bijector as bijector_lib

# Modified from tensorflow_probability/python/bijectors/real_nvp.py
class RealNVP(bijector_lib.Bijector):
  def __init__(self,
               num_masked,
               shift_and_log_scale_fn=None,
               bijector_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               name=None):
    name = name or 'real_nvp'
    if num_masked < 0:
      raise ValueError('num_masked must be a non-negative integer.')
    self._num_masked = num_masked
    # At construction time, we don't know input_depth.
    self._input_depth = None
    if bool(shift_and_log_scale_fn) == bool(bijector_fn):
      raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                       '`bijector_fn` should be specified.')
    if shift_and_log_scale_fn:
      def _bijector_fn(x0, input_depth, **condition_kwargs):
        shift, log_scale = shift_and_log_scale_fn(x0, input_depth,
                                                  **condition_kwargs)
        # ** First modification is here.
        return affine_scalar.AffineScalar(shift=shift, scale=log_scale)

      bijector_fn = _bijector_fn

    if validate_args:
      bijector_fn = _validate_bijector_fn(bijector_fn)

    # Still do this assignment for variable tracking.
    self._shift_and_log_scale_fn = shift_and_log_scale_fn
    self._bijector_fn = bijector_fn

    super(RealNVP, self).__init__(
        forward_min_event_ndims=1,
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)

  def _cache_input_depth(self, x):
    if self._input_depth is None:
      self._input_depth = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
      if self._input_depth is None:
        raise NotImplementedError(
            'Rightmost dimension must be known prior to graph execution.')
      if self._num_masked >= self._input_depth:
        raise ValueError(
            'Number of masked units must be smaller than the event size.')

  def _forward(self, x, **condition_kwargs):
    self._cache_input_depth(x)
    x0, x1 = x[..., :self._num_masked], x[..., self._num_masked:]
    y1 = self._bijector_fn(x0, self._input_depth - self._num_masked,
                           **condition_kwargs).forward(x1)
    y = tf.concat([x0, y1], axis=-1)
    return y

  def _inverse(self, y, **condition_kwargs):
    self._cache_input_depth(y)
    y0, y1 = y[..., :self._num_masked], y[..., self._num_masked:]
    x1 = self._bijector_fn(y0, self._input_depth - self._num_masked,
                           **condition_kwargs).inverse(y1)
    x = tf.concat([y0, x1], axis=-1)
    return x

  def _forward_log_det_jacobian(self, x, **condition_kwargs):
    self._cache_input_depth(x)
    x0, x1 = x[..., :self._num_masked], x[..., self._num_masked:]
    return self._bijector_fn(x0, self._input_depth - self._num_masked,
                             **condition_kwargs).forward_log_det_jacobian(
                                 x1, event_ndims=1)

  def _inverse_log_det_jacobian(self, y, **condition_kwargs):
    self._cache_input_depth(y)
    y0, y1 = y[..., :self._num_masked], y[..., self._num_masked:]
    return self._bijector_fn(y0, self._input_depth - self._num_masked,
                             **condition_kwargs).inverse_log_det_jacobian(
                                 y1, event_ndims=1)

def real_nvp_default_template(hidden_layers,
                              shift_only=False,
                              activation=tf.nn.relu,
                              name=None,
                              *args,  # pylint: disable=keyword-arg-before-vararg
                              **kwargs):
  with tf.compat.v1.name_scope(name or 'real_nvp_default_template'):

    def _fn(x, output_units, **condition_kwargs):
      """Fully connected MLP parameterized via `real_nvp_template`."""
      if condition_kwargs:
        raise NotImplementedError(
            'Conditioning not implemented in the default template.')

      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
        reshape_output = lambda x: x[0]
      else:
        reshape_output = lambda x: x
      for units in hidden_layers:
        x = tf1.layers.dense(
            inputs=x,
            units=units,
            activation=activation,
            *args,  # pylint: disable=keyword-arg-before-vararg
            **kwargs)
      x = tf1.layers.dense(
          inputs=x,
          units=(1 if shift_only else 2) * output_units,
          activation=None,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)
      if shift_only:
        return reshape_output(x), None
      shift, log_scale = tf.split(x, 2, axis=-1)
       # ** Here is the second modification.
      return reshape_output(shift), 1e-7 + tf.nn.softplus(reshape_output(log_scale))

    return tf1.make_template('real_nvp_default_template', _fn)

class RealNVPBijector(tf.keras.Model):
  def __init__(self, dimensions, n_couplings, hidden_layers, dense_kwargs):
    super(RealNVPBijector, self).__init__()
    permutations = [np.random.permutation(dimensions)
                    for _ in range(n_couplings)]
    bijectors = []
    for permutation in permutations:
      bijectors.append(RealNVP(
        dimensions // 2,
        real_nvp_default_template(hidden_layers, **dense_kwargs)))
      bijectors.append(tfb.Permute(permutation))
    self._bijector = tfb.Chain(bijectors)

  def call(self, inputs):
    return self._bijector.forward(inputs)

class InnerProdCritic(tf.keras.Model):
  def call(self, x, y):
    return tf.matmul(x, y, transpose_b=True)

class BilinearCritic(tf.keras.Model):
  def __init__(self, feature_dim=100, **kwargs):
    super(BilinearCritic, self).__init__(**kwargs)
    self._W = tfkl.Dense(feature_dim, use_bias=False)

  def call(self, x, y):
    return tf.matmul(x, self._W(y), transpose_b=True)

# Copied from
# https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
class ConcatCritic(tf.keras.Model):
  def __init__(self, hidden_dim=200, layers=1, activation='relu', **kwargs):
    super(ConcatCritic, self).__init__(**kwargs)
    # output is scalar score
    self._f = MLP([hidden_dim for _ in range(layers)]+[1], False, {"activation": "relu"})

  def call(self, x, y):
    batch_size = tf.shape(input=x)[0]
    # Tile all possible combinations of x and y
    x_tiled = tf.tile(x[None, :],  (batch_size, 1, 1))
    y_tiled = tf.tile(y[:, None],  (1, batch_size, 1))
    # xy is [batch_size * batch_size, x_dim + y_dim]
    xy_pairs = tf.reshape(tf.concat((x_tiled, y_tiled), axis=2),
                          [batch_size * batch_size, -1])
    # Compute scores for each x_i, y_j pair.
    scores = self._f(xy_pairs) 
    return tf.transpose(a=tf.reshape(scores, [batch_size, batch_size]))


class SeparableCritic(tf.keras.Model):
  def __init__(self, hidden_dim=100, output_dim=100, layers=1,
               activation='relu', **kwargs):
    super(SeparableCritic, self).__init__(**kwargs)
    self._f_x = MLP([hidden_dim for _ in range(layers)] + [output_dim], False, {"activation": activation})
    self._f_y = MLP([hidden_dim for _ in range(layers)] + [output_dim], False, {"activation": activation})

  def call(self, x, y):
    x_mapped = self._f_x(x)
    y_mapped = self._f_y(y)
    return tf.matmul(x_mapped, y_mapped, transpose_b=True)

def train(g1,
          g2,
          critic,
          loss_fn,
          learning_rate,
          batch_size=TRAIN_BATCH_SIZE,
          n_iters=15000,
          n_evals=15,
          compute_jacobian=False,
          noise_std=0.0,
          data_dimensions=DIMS//2,
          n_iter=1,
          loss_name='InfoNCE',
          ):
  """Runs the training loop for a fixed model.

  Args:
    g1: Function, maps input1 to representation.
    g2: Function, maps input2 to representation.
    critic: Function, maps two representations to scalar.
    loss_fn: Function, mutual information estimator.
    learning_rate: Learning rate.
    batch_size: Training batch size.
    n_iters: Number of optimization iterations.
    n_evals: Number of model evaluations.
    compute_jacobian: Whether to estimate the singular values of the Jacobian.
    noise_std: Standard deviation for the Gaussian noise. Default is 0.0.
    data_dimensions: The dimension of the data. By default it's half of the
      original data dimension.
  Returns:
    Returns and instance of `Results` tuple.
  """
  x_1, x_2, _ = processed_train_data(data_dimensions, batch_size)

  if noise_std > 0.0:
    assert x_1.shape == x_2.shape, "X1 and X2 shapes must agree to add noise!"
    noise = noise_std * tf.random.normal(x_1.shape)
    x_1 += noise
    x_2 += noise

  # Compute the representations.
  code_1, code_2 = g1(x_1), g2(x_2)
  critic_matrix = critic(code_1, code_2)
  # Compute the Jacobian of g1 if needed.
  if compute_jacobian:
    jacobian = gradients.batch_jacobian(code_1, x_1, use_pfor=False)
    singular_values = tf.linalg.svd(jacobian, compute_uv=False)

  # Optimizer setup.
  loss = loss_fn(critic_matrix)
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
  if not loss_name == 'wpc':
      optimizer_op = optimizer.minimize(loss)
  else:
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    optimizer_op = optimizer.apply_gradients(capped_gvs)

  with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())

    # Subgraph for eval (add noise to input if necessary)
    data_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, data_dimensions])
    data_ph_noisy = data_ph + noise_std * tf.random.normal(tf.shape(input=data_ph))
    codes = g1(data_ph_noisy)

    training_losses, testing_losses, classification_accuracies, iters, sigmas \
      = [], [], [], [], []
    # Main training loop.
    for iter_n in range(n_iters):
      # Evaluate the model performance.
      if iter_n % (n_iters // n_evals) == 0:
        iters.append(iter_n)
        accuracy = get_classification_accuracy(session, codes, data_ph, data_dimensions)
        classification_accuracies.append(accuracy)
        testing_losses.append(
            get_testing_loss(x_test, session, loss, data_ph, data_dimensions))
        if compute_jacobian:
          sigmas.append(session.run(singular_values))
        print("{:d}th iter Loss_name {} Step {:>10d} fit {:>.5f} DS {} B {:d} lr {:f}".format(\
                n_iter, loss_name, iter_n, accuracy, args.dataset, args.batch_size, args.lr))
      # Run one optimization step.
      loss_np, _ = session.run([loss, optimizer_op])
      training_losses.append(loss_np)

  return Results(iterations=iters,
                 training_losses=training_losses,
                 testing_losses=testing_losses,
                 classification_accuracies=classification_accuracies,
                 singular_values=sigmas)


def run_sweep(nets, critics, loss_fns, exp_name, **kwargs):
  """Runs the sweep across encoder networks, critics, and the estimators."""
  grid = itertools.product(nets, critics, loss_fns)
  data_frames = []
  results_with_singular_values = []
  for nets_name, critic_name, loss_name in grid:
    print("[New experiment] encoder: {}, critic: {}, loss: {}".format(
        nets_name, critic_name, loss_name))
    with tf.Graph().as_default():
      g1, g2 = nets[nets_name]()
      critic = critics[critic_name]()
      loss_fn = loss_fns[loss_name]
      results_per_run = []
      for n in range(NRUNS):
        try:
          print("{:d}th run, loss: {}".format(n, loss_name))
          if loss_name == "drfc" and TFDS_NAME == "cifar10":
            results = train(g1, g2, critic, loss_fn, **kwargs, learning_rate=LEARNING_RATE, n_iter=n, loss_name=loss_name)
            #results = train(g1, g2, critic, loss_fn, **kwargs, learning_rate=1e-4, n_iter=n, loss_name=loss_name)
          else:
            results = train(g1, g2, critic, loss_fn, **kwargs, learning_rate=LEARNING_RATE, n_iter=n, loss_name=loss_name)
          results_per_run.append(results)
        except Exception as ex:
          print("Run {} failed! Error: {}".format(n, ex))
      for i, result in enumerate(results_per_run):
        data_frames.append(convert_to_data_frame(
            result, exp_name, nets_name, critic_name, loss_name, i))
      if kwargs.get('compute_jacobian', False):
        results_with_singular_values.append((
            ResultsConfig(nets_name, critic_name, loss_name), results_per_run
        ))
  
  return {
      "df": pd.concat(data_frames), 
      "singular_values": results_with_singular_values
  }

#@title Run experiment or load precomputed results { display-mode: "form" }
def run_all_experiments():
  tf.compat.v1.reset_default_graph()
  wpc_loss = lambda x: -infonce_lower_bound(x)
  cpc_loss = lambda x: -infonce_lower_bound(x)
  #nwj_loss = lambda x: -nwj_lower_bound(x)
  drfc_loss = lambda x: -our_lower_bound(x)
  pcc_loss = lambda x: -js_fgan_lower_bound(x)
  loss_fcts = {
      "wpc": wpc_loss,
      "pcc": pcc_loss,
      "drfc": drfc_loss,
      #"nwj": nwj_loss,
      "cpc": cpc_loss,
      }
  kwargs = dict(
      shift_only=True,
      activation=lambda x: tf.nn.relu(x),
      kernel_initializer=tf.compat.v1.initializers.truncated_normal(stddev=0.0001),
      bias_initializer='zeros')
  nets = {
      "realnvp": lambda: (
          RealNVPBijector(DIMS // 2, n_couplings=30, hidden_layers=[512, 512], dense_kwargs=kwargs),
          RealNVPBijector(DIMS // 2, n_couplings=30, hidden_layers=[512, 512], dense_kwargs=kwargs)
          )
      }
  critics = {
      "bilinear": lambda: BilinearCritic(feature_dim=DIMS//2),
  }
  return run_sweep(nets, critics, loss_fcts, "invertible", n_iters=21000, n_evals=21)

if RUN_EXPERIMENTS:
  data_invertible = run_all_experiments()["df"]
  data_invertible.to_pickle(RESULT_DIR)
else:
  os.system("wget -q -N https://storage.googleapis.com/mi_for_rl_files/mi_results.pkl")
  data_invertible = pd.read_pickle('mi_results.pkl')
  data_invertible = data_invertible[data_invertible.exp_name == "invertible"]
