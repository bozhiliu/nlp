apt-get update
apt-get install -qq build-essential python-pip cmake mercurial git emacs wget unzip
pip install --upgrade==9.0.1
pip install scikit-learn numpy scipy matplotlib ipython jupyter pandas sympy nose nltk tensorflow
cd
git clone https://github.com/bozhiliu/nlp.git
cd nlp
tar -xzf ebm_nlp_1_00.tar.gz
cd acl_scripts/lstm-crf/
make glove

