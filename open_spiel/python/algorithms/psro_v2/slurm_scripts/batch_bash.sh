#!/usr/bin/env bash

for file in ./scripts_markov/*
do
  sbatch "$file"
  sleep 2
done
