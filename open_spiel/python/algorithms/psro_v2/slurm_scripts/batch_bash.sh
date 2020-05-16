#!/usr/bin/env bash

for file in ./scripts_laser_tag/*
do
  sbatch "$file"
  sleep 2
done
