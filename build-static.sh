#!/bin/bash
set -ex
docker build -t tahoe-static-build -f Dockerfile.pyinstaller-static .
docker create --name=tahoe-static-build tahoe-static-build arbitrary_string
mkdir -p build
docker cp tahoe-static-build:/tahoe-mount build
docker cp tahoe-static-build:/tahoe-mount-static build
docker cp tahoe-static-build:/tahoe-mount-bundle build
docker cp tahoe-static-build:/tahoe-upload build
docker cp tahoe-static-build:/tahoe-upload-static build
docker cp tahoe-static-build:/tahoe-upload-bundle build
docker container rm tahoe-static-build
