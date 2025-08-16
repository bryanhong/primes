#!/bin/bash -x

# This installs required packages for CUDA
# Usage: . ./venv.sh

python3 -m venv venv
echo 'export CUDA_ROOT=/usr/local/cuda' >> venv/bin/activate
echo 'export PATH=${CUDA_ROOT}/bin:${PATH}' >> venv/bin/activate
echo 'export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}' >> venv/bin/activate
source venv/bin/activate
pip3 install -r requirements.txt
