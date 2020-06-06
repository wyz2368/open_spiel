#!/usr/bin/env bash

for file in ./scripts_markov_soccer/*
do
  sbatch "$file"
  sleep 2
done
