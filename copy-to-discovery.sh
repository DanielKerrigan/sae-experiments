#!/usr/bin/env bash

rsync -av --exclude 'cache*.arrow' datasets kerrigan.d@xfer.discovery.neu.edu:/work/vis/users/kerrigan.d/mi/sae-experiments
