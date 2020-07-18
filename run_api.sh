#!bin/bash

docker image build -t python-system .
docker run --name python-sys-d -p 80:8080 python-system



