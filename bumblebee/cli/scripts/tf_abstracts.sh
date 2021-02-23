#!/bin/bash
# $1 is repository with scripts
# $2 is prefix
pigz -d data/${2}.gz -c | python3 ${1}/tf_abstracts.py tfrecords/$2
