#!/bin/sh
python /home/apsisdev/ansary/CODES/apsis-rec/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/refined/f.bn/ /home/apsisdev/ansary/DATASETS/Recognition/ f.bn ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/apsis-rec/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/refined/f.en/ /home/apsisdev/ansary/DATASETS/Recognition/ f.en ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/apsis-rec/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/refined/n.bn/ /home/apsisdev/ansary/DATASETS/Recognition/ n.bn ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/apsis-rec/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/refined/n.en/ /home/apsisdev/ansary/DATASETS/Recognition/ n.en ROBUSTSCANNER
echo succeeded