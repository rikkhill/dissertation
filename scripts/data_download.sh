#!/usr/bin/env bash

# Get 1M and genome movielens dataset
echo "Downloading dataset"
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip -P /tmp/ml_data/
wget http://files.grouplens.org/datasets/tag-genome/tag-genome.zip -P /tmp/ml_data/

echo "Unzipping dataset"
unzip /tmp/ml_data/ml-1m.zip -d ./data/
unzip /tmp/ml_data/tag-genome.zip -d ./data/
mv ./data/ml-1m ./data/1M
mv ./data/tag-genome ./data/genome
rm -rf /tmp/ml_data/
