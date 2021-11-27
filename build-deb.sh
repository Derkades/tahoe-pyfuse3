#!/bin/bash
set -ex
docker build -t tahoe-mount-deb-builder -f Dockerfile.deb-build --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
ID=$(docker create tahoe-mount-deb-builder)
docker cp "$ID:/data/debs/." .
docker rm -v $ID
