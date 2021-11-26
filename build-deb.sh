#!/bin/bash
set -ex
docker build -t tahoe-deb-builder -f Dockerfile.deb .
docker run --rm -it --mount "type=bind,source=$(pwd),target=/data" --user "$(id -u)" tahoe-deb-builder
