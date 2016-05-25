#!/usr/bin/env bash

# Get 1M movielens dataset
echo "Downloading dataset"
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip -P /tmp/ml_data/
echo "Unzipping dataset"
unzip /tmp/ml_data/ml-1m.zip -d ./data/
mv ./data/ml-1m ./data/1M
rm -rf /tmp/ml_data/