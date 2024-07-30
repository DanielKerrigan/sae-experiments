sourceconda

conda create -n sae-experiments python=3.10

conda activate sae-experiments

export SKIP_JUPYTER_BUILDER=1

pip install -e ../saefarer

