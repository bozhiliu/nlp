FROM ubuntu:16.04

MAINTAINER bozhiliu@email.arizona.edu

RUN apt-get update
RUN apt-get install -y git
RUN git clone https://github.com/bozhiliu/nlp.git /root/bozhiliu
RUN /root/bozhiliu/process.sh