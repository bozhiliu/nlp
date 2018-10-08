# NLP: NER task using LSTM and adversarial learning

This is a fork of [EBM NLP repository](https://github.com/bepnye/EBM-NLP) with the following changes:

1. Fix one of the word embedding error
2. Adding a process.sh file that install necessary packages and run commands

## Usage:
1. Baseline: Download the Dockerfile and build image from it
..1. The Dockerfile builds from a Ubuntu image.
..2. Then the docker file install git
..3. This repo will be pulled and process.sh will be executed. This file specifies the packages and operations to take.
