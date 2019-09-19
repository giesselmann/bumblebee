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
    input_len = 500
    target_len = 50
    batch_size = 8
    d_input = 4096
    d_output = 6
    d_model = 512
    transformer_hparams = {'d_input' : d_input,
                           'd_output' : d_output,
                           'd_model' : d_model,
                           'dff' : 4096,
                           'num_heads' : 8,
                           'max_iterations' : 16,
                           'encoder_time_scale' : 10000,
                           'decoder_time_scale' : 1000}

    temp_input = tf.random.uniform((batch_size, input_len), 0, d_input-2, dtype=tf.int64)
    temp_target = tf.random.uniform((batch_size, target_len+1), 1, d_output-2, dtype=tf.int64)
    temp_input_len = tf.random.uniform((batch_size, 1), 5, input_len - 1, dtype=tf.int64)
    temp_target_len = tf.random.uniform((batch_size, 1), 5, target_len - 1, dtype=tf.int64)

    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    with strategy.scope():
        learning_rate = TransformerLRS(d_model, warmup_steps=2000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,amsgrad=True)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
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
            with tf.GradientTape() as tape:
                predictions = transformer(input, training=True)
                loss = loss_function(target, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_accuracy.update_state(target, predictions)
            return loss

        def test_step(inputs):
            input, target = inputs
            predictions = transformer(input, training=False)
            t_loss = loss_function(target, predictions)
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

    exit(0)

    transformer = tf.keras.utils.multi_gpu_model(transformer, 2)
    transformer.compile(optimizer=optimizer, loss=loss_function)
    print(transformer.metrics_names)
    loss = transformer.train_on_batch([temp_input, temp_target[:,:-1], temp_input_len, temp_target_len],
                                            temp_target[:,1:])
    print(loss)

    exit(0)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, input_len), dtype=tf.int64),
        tf.TensorSpec(shape=(None, target_len+1), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64)
        ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar, inp_len, tar_len):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions = transformer([inp, tar_inp, inp_len, tar_len], training=True)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in range(50):
        start = time.time()
        #train_loss.reset_states()
        train_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for batch in range(1000):
            train_step(temp_input, temp_target, temp_input_len, temp_target_len)
            #print("batch {}".format(batch))
            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                pass
            if (epoch + 1) % 5 == 0 and False:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                     ckpt_save_path))

                print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
