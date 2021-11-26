#!/bin/bash
set -ex
docker build -t tahoe-deb-builder -f Dockerfile.deb-build .
docker run --rm -it --mount "type=bind,source=$(pwd),target=/data" --user "$(id -u)" tahoe-deb-builder
