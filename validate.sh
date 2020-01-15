# make folder for tag
TAG=$1
mkdir -p $TAG

rsync promethion:/data/machine_learning/projects/bumblebee/training_checkpoints/bb/${TAG}/transformer/checkpoint ./${TAG}

CKPT=`cat ./${TAG}/checkpoint | head -1 | perl -lne 'print $& if /ckpt[-]\d{3}/'`
echo "Fetching" $CKPT "."
rsync promethion:/data/machine_learning/projects/bumblebee/training_checkpoints/bb/${TAG}/transformer/${CKPT}* ./${TAG}/
rsync promethion:/data/machine_learning/projects/bumblebee/training_configs/bb/${TAG}/hparams.yaml ./${TAG}/

source /project/miniondev/virtual/tf_cpu2/bin/activate
echo "Converting weights."
python3 bumblebee/bumblebee.py weights ./${TAG}/hparams.yaml ./${TAG}
echo "Validation on training data."
python3 bumblebee/bumblebee.py validate ./${TAG}/hparams.yaml ./${TAG}/weights.h5 ./FAK57924_4bfc0dd1d11f24f93af027273154e6e90bbed258_0.mm10.tfrec -t 64 --minibatch_size 16
deactivate
