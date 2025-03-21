#!/bin/bash

docker build -t transformer-api .
docker run -p 8000:8000 transformer-api
