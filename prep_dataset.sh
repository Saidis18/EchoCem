#!/usr/bin/env bash
# Usage: ./prep_dataset.sh [DATA_DIR]
# Default: DATA_DIR="./data"

set -e
DATA_DIR="${1:-./data}"

X_NAME="X_train_uDRk9z9"

mkdir -p "$DATA_DIR"/X_train_160
mkdir -p "$DATA_DIR"/X_train_272

cp "$DATA_DIR"/"$X_NAME"/images/well_{3,4}*.npy "$DATA_DIR"/X_train_160/
cp "$DATA_DIR"/"$X_NAME"/images/well_[^34]*.npy "$DATA_DIR"/X_train_272/

Y_NAME=$(ls "$DATA_DIR" | grep Y_train_)
cp "$DATA_DIR"/"$Y_NAME" "$DATA_DIR"/Y_train_160.csv
cp "$DATA_DIR"/"$Y_NAME" "$DATA_DIR"/Y_train_272.csv
sed -i -E '/^well_[^34]\w*/d' "$DATA_DIR"/Y_train_160.csv
sed -i -E '/^well_(3|4)\w*/d' "$DATA_DIR"/Y_train_272.csv
