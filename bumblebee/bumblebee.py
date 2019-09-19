# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : BumbleBee
#
#  DESCRIPTION   : Nanopore Basecalling
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019 Pay Giesselmann, Max Planck Institute for Molecular Genetics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Pay Giesselmann
# ---------------------------------------------------------------------------------
import os
import time
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tf_transformer import Transformer
from tf_transformer_util import TransformerLRS



# basic simulation
class pore_model():
    def __init__(self, model_file):
        def model_iter(iterable):
            for line in iterable:
                yield line.strip().split('\t')[:3]
        with open(model_file, 'r') as fp:
            model_dict = {x[0]:(float(x[1]), float(x[2])) for x in model_iter(fp)}
        self.kmer = len(next(iter(model_dict.keys())))
        self.model_median = np.median([x[0] for x in model_dict.values()])
        self.model_MAD = np.mean(np.absolute(np.subtract([x[0] for x in model_dict.values()], self.model_median)))
        self.model_values = np.array([x[0] for x in model_dict.values()])
        self.model_dict = model_dict

    def generate_signal(self, sequence, samples=10, noise=False):
        signal = []
        level_means = np.array([self.model_dict[kmer][0] for kmer in
            [sequence[i:i+self.kmer] for i in range(len(sequence)-self.kmer + 1)]])
        if samples and not noise:
            sig = np.repeat(level_means, samples)
        elif not noise:
            sig = np.repeat(level_means, np.random.uniform(6, 10, len(level_means)).astype(int))
        else:
            level_stdvs = np.array([self.model_dict[kmer][1] for kmer in
                [sequence[i:i+self.kmer] for i in range(len(sequence)-self.kmer + 1)]])
            level_samples = np.random.uniform(6, 10, len(level_means)).astype(int)
            level_means = np.repeat(level_means, level_samples)
            level_stdvs = np.repeat(level_stdvs, level_samples)
            sig = np.random.normal(level_means, 3 * level_stdvs)
        return sig

    def quantile_nrm(self, signal_raw, q=30):
        base_q = np.quantile(self.model_values, np.linspace(0,1,q))
        raw_q = np.quantile(signal_raw, np.linspace(0,1,q))
        p = np.poly1d(np.polyfit(raw_q, base_q, 3))
        return (p(signal_raw) - self.model_median) / self.model_MAD




if __name__ == '__main__':
    input_max_len = 2000
    target_max_len = 200
    batch_size = 32
    d_input = 4096
    d_output = 6
    d_model = 128
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    transformer_hparams = {'d_input' : d_input,
                           'd_output' : d_output,
                           'd_model' : d_model,
                           'dff' : d_model * 4,
                           'num_heads' : 8,
                           'max_iterations' : 16,
                           'encoder_time_scale' : 10000,
                           'decoder_time_scale' : 1000,
                           'input_memory_comp' : 8,
                           'target_memory_comp' : 4}

    temp_input = tf.random.uniform((batch_size, input_max_len), 0, d_input-2, dtype=tf.int64)
    temp_target = tf.random.uniform((batch_size, target_max_len+1), 0, d_output-2, dtype=tf.int64)
    temp_input_len = tf.random.uniform((batch_size, 1), 5, input_max_len - 1, dtype=tf.int64)
    temp_target_len = tf.random.uniform((batch_size, 1), 5, target_max_len - 1, dtype=tf.int64)

    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0'])
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    with strategy.scope():
        learning_rate = TransformerLRS(d_model, warmup_steps=2000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,amsgrad=True)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        def loss_function(real, pred, target_lengths):
            mask = tf.sequence_mask(target_lengths, target_max_len, dtype=tf.float32)
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.nn.compute_average_loss(loss_, global_batch_size=batch_size)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        transformer = Transformer(hparams=transformer_hparams)
        #transformer.summary()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer)

        def train_step(inputs):
            input, target = inputs
            target_lengths = input[3]
            with tf.GradientTape() as tape:
                predictions = transformer(input, training=True)
                loss = loss_function(target, predictions, target_lengths)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_accuracy.update_state(target, predictions)
            return loss

        def test_step(inputs):
            input, target = inputs
            target_lengths = input[3]
            predictions = transformer(input, training=False)
            t_loss = loss_function(target, predictions, target_lengths)
            test_loss.update_state(t_loss)
            test_accuracy.update_state(target, predictions)

        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

        for epoch in range(10):
            total_loss = 0.0
            num_batches = 0
            for batch in tqdm(range(100), desc='Training'):
                total_loss += distributed_train_step((
                        [temp_input, temp_target[:,:-1], temp_input_len, temp_target_len],
                        temp_target[:,1:]))
                num_batches += 1
            train_loss = total_loss / num_batches
            for batch in tqdm(range(10), desc='Testing'):
                distributed_test_step((
                        [temp_input, temp_target[:,:-1], temp_input_len, temp_target_len],
                        temp_target[:,1:]))

            print("Epoch {}: train loss: {}; test loss: {}; accuracy: {}".format(epoch,
                        train_loss,
                        test_loss.result(),
                        test_accuracy.result()))
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
