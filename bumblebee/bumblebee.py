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
    config = tf.compat.v1.ConfigProto()
    gpus = [0,1]
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    config.gpu_options.visible_device_list = ','.join([str(n) for n in gpus])
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8
    tf.compat.v1.Session(config=config)
    d_input = 4096
    d_output = 6
    d_model = 512
    dff = 4096
    max_iterations = 16
    num_heads = 8
    sig_len = 1000
    seq_len = 100
    batch_size = 16
    max_timescale = 50

    temp_input = tf.random.uniform((batch_size, sig_len), 0, d_input-2, dtype=tf.int64)
    temp_target = tf.random.uniform((batch_size, seq_len+1), 0, d_output-2, dtype=tf.int64)
    temp_input_len = tf.random.uniform((batch_size, 1), 0, sig_len - 1, dtype=tf.int64)
    temp_target_len = tf.random.uniform((batch_size, 1), 0, seq_len - 1, dtype=tf.int64)

    #tf_out = transformer((temp_input, temp_target, temp_input_len, temp_target_len), training=False)
    #transformer.summary()
    #transformer = tf.keras.utils.multi_gpu_model(transformer, 2, cpu_relocation=True)

    learning_rate = TransformerLRS(d_model, warmup_steps=2000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,amsgrad=True)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    checkpoint_path = "./checkpoints/train"
    with tf.device("cpu:0"):
        transformer = Transformer(d_input, d_model, d_output,
                                sig_len, seq_len,
                                max_iterations=max_iterations, num_heads=num_heads, dff=dff)
    #transformer = tf.keras.utils.multi_gpu_model(transformer, 2)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, sig_len), dtype=tf.int64),
        tf.TensorSpec(shape=(None, seq_len+1), dtype=tf.int64),
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
        train_loss.reset_states()
        train_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for batch in range(500):
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
