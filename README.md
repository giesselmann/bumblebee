# bumblebee
Nanopore Basecalling

# on bb8
tensorboard --logdir ./ --host=127.0.0.1 --port 6006

# @home
ssh -L 127.0.0.1:6006:127.0.0.1:6006 bb8
