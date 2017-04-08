#!/bin/bash

kitti_root_dir=$1

if [ -z "$kitti_root_dir" ]; then
    echo "Usage: create_label.sh <kitti-root-dir> (e.g. ~/data/kitti/object/training)"
    exit 1
fi

echo "Generating labels for kitti..."
python kitti/kitti.py $kitti_root_dir

echo "Generating labels for kitti with proj..."
python kitti_proj/kitti.py $kitti_root_dir

echo "Generating labels for variants ..."
variants=(miami carla)

for variant in ${variants[@]}; do
    mkdir -p $variant
    cp -R kitti/window_files $variant
    sed -i -e "s/image_2/image_2_$variant/g" $variant/window_files/*

    mkdir -p ${variant}_proj
    cp -R kitti_proj/window_files ${variant}_proj
    sed -i -e "s/image_2/image_2_${variant}/g" ${variant}_proj/window_files/*
done

mkdir -p mixed/window_files mixed_proj/window_files
for window_file in $(ls miami/window_files); do
    cat miami/window_files/$window_file carla/window_files/$window_file > mixed/window_files/$window_file
    cat miami_proj/window_files/$window_file carla_proj/window_files/$window_file > mixed_proj/window_files/$window_file
done

