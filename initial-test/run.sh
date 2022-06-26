#!/bin/bash

tar -xvf data.tar.xz

python ../../hypercube-plotter/hyperplotter.py  -p ./data/params/ -c ./config.yml -i ./plots.yml -d ./data/data/ -o ./
