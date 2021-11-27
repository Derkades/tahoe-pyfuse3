#!/bin/bash
set -ex
docker build -t derkades/tahoe-mount -f Dockerfile .
