# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions and classes related to optimization (weight updates).
Modified from the original BERT code to allow for having separate learning
rates for different layers of the network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import tensorflow.compat.v1 as tf


def create_optimizer(
        loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
        warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
        layerwise_lr_decay_power=-1, n_transformer_layers=None, name="adamw", var_map=None):
    """Creates an optimizer and training op."""
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=lr_decay_power,
        cycle=False)
    warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
    learning_rate *= tf.minimum(
        1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

    if layerwise_lr_decay_power > 0:
        learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power,
                                       n_transformer_layers)
    if name == "recadam":
        optimizer = RecAdamOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            anneal_k=0.5,
            anneal_t0=250,
            anneal_w=1.0,
            pretrain_cof=5000.0,
            pretrain_params=var_map
        )
    else:
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    if use_tpu:
        optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _apply_gradients(self, grads_and_vars, learning_rate):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self.weight_decay_rate > 0:
                if self._do_use_weight_decay(param_name):
                    update += self.weight_decay_rate * param

            # layerwise lr
            # lr_weight = 1.0
            # if "/dependence/" in param_name:
            #     print(param_name)
            #     lr_weight = 5.0
            update_with_lr = learning_rate * update
            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])

        return assignments

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if isinstance(self.learning_rate, dict):
            key_to_grads_and_vars = {}
            for grad, var in grads_and_vars:
                update_for_var = False
                for key in self.learning_rate:
                    if key in var.name:
                        update_for_var = True
                        if key not in key_to_grads_and_vars:
                            key_to_grads_and_vars[key] = []
                        key_to_grads_and_vars[key].append((grad, var))
                if not update_for_var:
                    raise ValueError("No learning rate specified for variable", var)
            assignments = []
            for key, key_grads_and_vars in key_to_grads_and_vars.items():
                assignments += self._apply_gradients(key_grads_and_vars,
                                                     self.learning_rate[key])
        else:
            assignments = self._apply_gradients(grads_and_vars, self.learning_rate)
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def _get_layer_lrs(learning_rate, layer_decay, n_layers):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = collections.OrderedDict({
        "/embeddings/": 0,
        "/embeddings_project/": 0,
        "task_specific/": n_layers + 2,
    })
    for layer in range(n_layers):
        key_to_depths["encoder/layer_" + str(layer) + "/"] = layer + 1
    return {
        key: learning_rate * (layer_decay ** (n_layers + 2 - depth))
        for key, depth in key_to_depths.items()
    }


"""RecAdam optimizer"""


def anneal_function(function, step, k, t0, weight):
    if function == 'sigmoid':
        return (1 / (1 + tf.exp(-k * tf.to_float(step - t0)))) * weight
    elif function == 'linear':
        return tf.minimum(1, step / t0) * weight
    elif function == 'constant':
        return weight
    else:
        raise ValueError


class RecAdamOptimizer(AdamWeightDecayOptimizer):
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="RecAdamOptimizer",
                 anneal_fun='sigmoid',
                 anneal_k=0.5,
                 anneal_t0=250,
                 anneal_w=1.0,
                 pretrain_cof=5000.0,
                 pretrain_params=None
                 ):
        # anneal_k (float): a hyperparam for the anneal function, decide the slop of the curve. Choice: [0.05, 0.1, 0.2, 0.5, 1]
        # anneal_t0 (float): a hyperparam for the anneal function, decide the middle point of the curve. Choice: [100, 250, 500, 1000]
        # anneal_w (float): a hyperparam for the anneal function, decide the scale of the curve. Default 1.0.
        # pretrain_cof (float): the coefficient of the quadratic penalty. Default 5000.0.
        # pretrain_params (list of tensors): the corresponding group of params in the pretrained model.
        super(RecAdamOptimizer, self).__init__(learning_rate, weight_decay_rate, beta_1, beta_2, epsilon,
                                               exclude_from_weight_decay, name)
        self.anneal_fun = anneal_fun
        self.anneal_k = anneal_k
        self.anneal_t0 = anneal_t0
        self.anneal_w = anneal_w
        self.pretrain_cof = pretrain_cof
        self.pretrain_params = pretrain_params

    def _apply_gradients(self, grads_and_vars, learning_rate, global_step):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            anneal_lambda = anneal_function(self.anneal_fun, global_step, self.anneal_k,
                                            self.anneal_t0, self.anneal_w)
            if param_name in self.pretrain_params:
                print(param)
                print(self.pretrain_params[param_name].shape)
                print("before:", update)
                print(param.dt)
                update = anneal_lambda * update + (self.anneal_w - anneal_lambda) * self.pretrain_cof * \
                         tf.to_float(param - tf.to_float(self.pretrain_params[param_name]))
                # update = anneal_lambda * update + (self.anneal_w - anneal_lambda) * self.pretrain_cof
                print("after:", update)

            if self.weight_decay_rate > 0:
                if self._do_use_weight_decay(param_name):
                    update += self.weight_decay_rate * param

            update_with_lr = learning_rate * update
            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])

        return assignments

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if isinstance(self.learning_rate, dict):
            key_to_grads_and_vars = {}
            for grad, var in grads_and_vars:
                update_for_var = False
                for key in self.learning_rate:
                    if key in var.name:
                        update_for_var = True
                        if key not in key_to_grads_and_vars:
                            key_to_grads_and_vars[key] = []
                        key_to_grads_and_vars[key].append((grad, var))
                if not update_for_var:
                    raise ValueError("No learning rate specified for variable", var)
            assignments = []
            for key, key_grads_and_vars in key_to_grads_and_vars.items():
                assignments += self._apply_gradients(key_grads_and_vars,
                                                     self.learning_rate[key], global_step)
        else:
            assignments = self._apply_gradients(grads_and_vars, self.learning_rate, global_step)
        return tf.group(*assignments, name=name)


# import logging
# import math
# import numpy as np
#
# import torch
# from torch.optim import Optimizer
#
#
# logger = logging.getLogger(__name__)
#
#
# def anneal_function(function, step, k, t0, weight):
#     if function == 'sigmoid':
#         return float(1 / (1 + np.exp(-k * (step - t0)))) * weight
#     elif function == 'linear':
#         return min(1, step / t0) * weight
#     elif function == 'constant':
#         return weight
#     else:
#         ValueError
#
#
# class RecAdam(Optimizer):
#     """ Implementation of RecAdam optimizer, a variant of Adam optimizer.
#     Parameters:
#         lr (float): learning rate. Default 1e-3.
#         betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
#         eps (float): Adams epsilon. Default: 1e-6
#         weight_decay (float): Weight decay. Default: 0.0
#         correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
#         anneal_fun (str): a hyperparam for the anneal function, decide the function of the curve. Default 'sigmoid'.
#         anneal_k (float): a hyperparam for the anneal function, decide the slop of the curve. Choice: [0.05, 0.1, 0.2, 0.5, 1]
#         anneal_t0 (float): a hyperparam for the anneal function, decide the middle point of the curve. Choice: [100, 250, 500, 1000]
#         anneal_w (float): a hyperparam for the anneal function, decide the scale of the curve. Default 1.0.
#         pretrain_cof (float): the coefficient of the quadratic penalty. Default 5000.0.
#         pretrain_params (list of tensors): the corresponding group of params in the pretrained model.
#     """
#
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True,
#                  anneal_fun='sigmoid', anneal_k=0, anneal_t0=0, anneal_w=1.0, pretrain_cof=5000.0, pretrain_params=None):
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
#                         anneal_fun=anneal_fun, anneal_k=anneal_k, anneal_t0=anneal_t0, anneal_w=anneal_w,
#                         pretrain_cof=pretrain_cof, pretrain_params=pretrain_params)
#         super().__init__(params, defaults)
#
#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#         for group in self.param_groups:
#             for p, pp in zip(group["params"], group["pretrain_params"]):
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
#
#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state["step"] = 0
#                     # Exponential moving average of gradient values
#                     state["exp_avg"] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(p.data)
#
#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]
#
#                 state["step"] += 1
#
#                 # Decay the first and second moment running average coefficient
#                 # In-place operations to update the averages at the same time
#                 exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
#                 exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
#                 denom = exp_avg_sq.sqrt().add_(group["eps"])
#
#                 step_size = group["lr"]
#                 if group["correct_bias"]:  # No bias correction for Bert
#                     bias_correction1 = 1.0 - beta1 ** state["step"]
#                     bias_correction2 = 1.0 - beta2 ** state["step"]
#                     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
#
#                 # With RecAdam method, the optimization objective is
#                 # Loss = lambda(t)*Loss_T + (1-lambda(t))*Loss_S
#                 # Loss = lambda(t)*Loss_T + (1-lambda(t))*\gamma/2*\sum((\theta_i-\theta_i^*)^2)
#                 if group['anneal_w'] > 0.0:
#                     # We calculate the lambda as the annealing function
#                     anneal_lambda = anneal_function(group['anneal_fun'], state["step"], group['anneal_k'],
#                                                     group['anneal_t0'], group['anneal_w'])
#                     assert anneal_lambda <= group['anneal_w']
#                     # The loss of the target task is multiplied by lambda(t)
#                     p.data.addcdiv_(-step_size * anneal_lambda, exp_avg, denom)
#                     # Add the quadratic penalty to simulate the pretraining tasks
#                     p.data.add_(-group["lr"] * (group['anneal_w'] - anneal_lambda) * group["pretrain_cof"], p.data - pp.data)
#                 else:
#                     p.data.addcdiv_(-step_size, exp_avg, denom)
#
#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 # Add weight decay at the end (fixed version)
#                 if group["weight_decay"] > 0.0:
#                     p.data.add_(-group["lr"] * group["weight_decay"], p.data)
#
#         return loss