#!/bin/bash
set -e

IMAGE="derkades/tahoe-mount"
MAJOR="1"
MINOR="2"
PATCH="0"

docker build \
    -t "$IMAGE" \
    -t "$IMAGE:$MAJOR" \
    -t "$IMAGE:$MAJOR.$MINOR" \
    -t "$IMAGE:$MAJOR.$MINOR.$PATCH" \
    -f Dockerfile .

if [ "$1" == "push" ]
then
    docker push "$IMAGE"
    docker push "$IMAGE:$MAJOR"
    docker push "$IMAGE:$MAJOR.$MINOR"
    docker push "$IMAGE:$MAJOR.$MINOR.$PATCH"
fi
