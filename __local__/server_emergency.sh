#!/bin/sh
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/fix/bn.ffd/ /home/apsisdev/ansary/DATASETS/Recognition/ bn.ffd ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/fix/bn.ffn/ /home/apsisdev/ansary/DATASETS/Recognition/ bn.ffn ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/fix/en.ffd/ /home/apsisdev/ansary/DATASETS/Recognition/ en.ffd ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/fix/en.ffn/ /home/apsisdev/ansary/DATASETS/Recognition/ en.ffn ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/fix/bn.nums/ /home/apsisdev/ansary/DATASETS/Recognition/ bn.nums ROBUSTSCANNER 
echo succeeded
