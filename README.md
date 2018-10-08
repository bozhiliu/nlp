# NLP: NER task using LSTM and adversarial learning

This is a fork of [EBM NLP repository](https://github.com/bepnye/EBM-NLP) with the following changes:

1. Fix one of the word embedding error
2. Adding a process.sh file that install necessary packages and run commands

## Usage:
### Baseline system
  Follow the instructions to download the Dockerfile and build image from it
  1. The Dockerfile builds from a Ubuntu 16.04 image.
  2. Then the docker file install git
  3. This repo will be pulled and process.sh will be executed. This file specifies the packages and operations to take. It will also download the word embeddings which will take time.
  4. Attach to docker to run the following:
      1. cd ~/EBM-NLP/acl_scripts/lstm-crf
      2. python train.py
      3. python evaulate.py
    
