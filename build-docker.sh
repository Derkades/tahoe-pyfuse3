#!/bin/bash
set -ex
docker build -t derkades/tahoe-mount -f Dockerfile.mount .
docker build -t derkades/tahoe-upload -f Dockerfile.upload .
