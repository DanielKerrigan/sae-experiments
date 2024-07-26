#!/usr/bin/env bash

rsync -av --exclude 'cache*.arrow' datasets models kerrigan.d@xfer.discovery.neu.edu:/scratch/kerrigan.d/mi/sae-experiments
