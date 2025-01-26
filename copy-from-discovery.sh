#!/usr/bin/env bash

root=/work/vis/users/kerrigan.d/mi/sae-experiments
rsync -av --exclude 'cache*.arrow' \
    "kerrigan.d@xfer.discovery.neu.edu:$root/saes" ":$root/datasets" .
