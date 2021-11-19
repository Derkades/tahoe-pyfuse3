#!/bin/bash
set -e
VERSION=0.0.0-2

rm -rf build/*_amd64
rm -rf build/*.deb

mkdir -p "build/tahoe-mount_${VERSION}_amd64/DEBIAN"
mkdir -p "build/tahoe-mount_${VERSION}_amd64/usr/sbin"

cat << EOF > "build/tahoe-mount_${VERSION}_amd64/DEBIAN/control"
Package: tahoe-mount
Version: $VERSION
Section: base
Priority: optional
Architecture: amd64
Maintainer: Robin Slot <robin@rkslot.nl>
Description: FUSE mount client for Tahoe-LAFS
EOF

cp build/mount.tahoe "build/tahoe-mount_${VERSION}_amd64/usr/sbin"
chmod +rx "build/tahoe-mount_${VERSION}_amd64/usr/sbin/mount.tahoe"

dpkg-deb --build --root-owner-group "build/tahoe-mount_${VERSION}_amd64"

mkdir -p "build/tahoe-upload_${VERSION}_amd64/DEBIAN"
mkdir -p "build/tahoe-upload_${VERSION}_amd64/usr/bin"

cat << EOF > "build/tahoe-upload_${VERSION}_amd64/DEBIAN/control"
Package: tahoe-upload
Version: $VERSION
Section: base
Priority: optional
Architecture: amd64
Maintainer: Robin Slot <robin@rkslot.nl>
Description: rsync-like upload program for Tahoe-LAFS
EOF

cp build/tahoe-upload "build/tahoe-upload_${VERSION}_amd64/usr/bin"
chmod +rx "build/tahoe-upload_${VERSION}_amd64/usr/bin/tahoe-upload"

dpkg-deb --build --root-owner-group "build/tahoe-upload_${VERSION}_amd64"
