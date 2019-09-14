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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt




def positional_encoding(seq_len, depth, d_model, max_timescale=10000):
    def get_angles(pos, j):
        angle_rates = 1 / np.power(max_timescale,
                                   (2 * (j//2)) / np.float32(d_model))
        return pos * angle_rates
    pos_rads = get_angles(np.arange(seq_len)[np.newaxis, :, np.newaxis],
                          np.arange(d_model)[np.newaxis, np.newaxis, :])
    depth_rads = get_angles(np.arange(depth)[:, np.newaxis, np.newaxis],
                            np.arange(d_model)[np.newaxis, np.newaxis, :])
    rads = pos_rads + depth_rads
    # apply sin to even indices in the array; 2i
    rads[:,:, 0::2] = np.sin(pos_rads[:, :, 0::2]) + np.sin(depth_rads[:, :, 0::2])
    # apply cos to odd indices in the array; 2i+1
    rads[:,:, 1::2] = np.cos(pos_rads[:, :, 1::2]) + np.cos(depth_rads[:, :, 1::2])
    pos_encoding = rads[np.newaxis, ...]
    # (1, depth, seq_len, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)




if __name__ == '__main__':
    tf.InteractiveSession()
    d_model = 32
    seq_len = 50
    max_timescale = 50
    pos_encoding = positional_encoding(seq_len, 8, d_model, max_timescale).eval()
    print(pos_encoding.shape)
    f, ax = plt.subplots(8)
    for i in range(8):
        ax[i].pcolormesh(pos_encoding[0][i], cmap='RdBu')
        ax[i].set_xlabel('')
        ax[i].set_xlim((0, d_model))
    #plt.colorbar()
    plt.show()
