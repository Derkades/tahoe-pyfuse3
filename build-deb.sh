#!/bin/bash
set -e

cd upload
dpkg-buildpackage -b --no-sign
cd ../mount
dpkg-buildpackage -b --no-sign
