#!/bin/bash
echo "Installing packages..."
apt-get install -qq -y build-essential python-pip cmake mercurial emacs wget unzip
pip install --upgrade pip==9.0.1
pip install scikit-learn numpy scipy matplotlib ipython jupyter pandas sympy nose nltk tensorflow logging

echo "Getting prebuilt Glove word embeddings"
cd ~
git clone https://github.com/bozhiliu/nlp.git EBM-NLP
cd EBM-NLP
tar -xzf ebm_nlp_1_00.tar.gz
cd acl_scripts/lstm-crf/
make glove

python ../../nltk_download.py

python build_data.py
