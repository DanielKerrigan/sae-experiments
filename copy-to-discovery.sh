#!/usr/bin/env bash

rsync -av --exclude 'cache*.arrow' datasets models kerrigan.d@xfer.discovery.neu.edu:/work/vis/users/kerrigan.d/mi/sae-experiments
