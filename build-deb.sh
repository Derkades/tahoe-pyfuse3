#!/bin/bash
set -e
VERSION=0.0.0-4

rm -rf build/*_amd64
rm -rf build/*.deb

# ------------------------ mount (static) ------------------------

mkdir -p "build/tahoe-mount-static_${VERSION}_amd64/DEBIAN"
mkdir -p "build/tahoe-mount-static_${VERSION}_amd64/usr/sbin"

STATIC_MOUNT_SIZE=$(stat -c%s build/mount.tahoe)
STATIC_MOUNT_SIZE=$((STATIC_MOUNT_SIZE/1024))
cat << EOF > "build/tahoe-mount-static_${VERSION}_amd64/DEBIAN/control"
Package: tahoe-mount-static
Version: $VERSION
Section: base
Priority: optional
Architecture: amd64
Maintainer: Robin Slot <robin@rkslot.nl>
Description: FUSE mount client for Tahoe-LAFS (static)
Installed-Size: $STATIC_MOUNT_SIZE
Conflicts: tahoe-mount
EOF

cp build/mount.tahoe "build/tahoe-mount-static_${VERSION}_amd64/usr/sbin"
chmod +rx "build/tahoe-mount-static_${VERSION}_amd64/usr/sbin/mount.tahoe"

dpkg-deb --build --root-owner-group "build/tahoe-mount-static_${VERSION}_amd64"

# ------------------------ mount ------------------------

mkdir -p "build/tahoe-mount_${VERSION}_amd64/DEBIAN"
mkdir -p "build/tahoe-mount_${VERSION}_amd64/usr/sbin"

MOUNT_SCRIPT_SIZE=$(stat -c%s mount.py)
MOUNT_SCRIPT_SIZE=$((MOUNT_SCRIPT_SIZE/1024))
cat << EOF > "build/tahoe-mount_${VERSION}_amd64/DEBIAN/control"
Package: tahoe-mount
Version: $VERSION
Section: base
Priority: optional
Architecture: amd64
Maintainer: Robin Slot <robin@rkslot.nl>
Description: FUSE mount client for Tahoe-LAFS
Installed-Size: $MOUNT_SCRIPT_SIZE
Conflicts: tahoe-mount-static
Depends: python3 (>= 3.7), python3-pyfuse3, python3-urllib3
EOF

cp mount.py "build/tahoe-mount_${VERSION}_amd64/usr/sbin/mount.tahoe"
chmod +rx "build/tahoe-mount_${VERSION}_amd64/usr/sbin/mount.tahoe"

dpkg-deb --build --root-owner-group "build/tahoe-mount_${VERSION}_amd64"

# ------------------------ tahoe-upload (static) ------------------------

mkdir -p "build/tahoe-upload-static_${VERSION}_amd64/DEBIAN"
mkdir -p "build/tahoe-upload-static_${VERSION}_amd64/usr/bin"

STATIC_UPLOAD_SIZE=$(stat -c%s build/tahoe-upload)
STATIC_UPLOAD_SIZE=$((STATIC_UPLOAD_SIZE/1024))
cat << EOF > "build/tahoe-upload-static_${VERSION}_amd64/DEBIAN/control"
Package: tahoe-upload-static
Version: $VERSION
Section: base
Priority: optional
Architecture: amd64
Maintainer: Robin Slot <robin@rkslot.nl>
Description: rsync-like upload program for Tahoe-LAFS (static)
Installed-Size: $STATIC_UPLOAD_SIZE
Conflicts: tahoe-upload
EOF

cp build/tahoe-upload "build/tahoe-upload-static_${VERSION}_amd64/usr/bin"
chmod +rx "build/tahoe-upload-static_${VERSION}_amd64/usr/bin/tahoe-upload"

dpkg-deb --build --root-owner-group "build/tahoe-upload-static_${VERSION}_amd64"

# ------------------------ tahoe-upload ------------------------

mkdir -p "build/tahoe-upload_${VERSION}_amd64/DEBIAN"
mkdir -p "build/tahoe-upload_${VERSION}_amd64/usr/bin"

UPLOAD_SCRIPT_SIZE=$(stat -c%s upload.py)
UPLOAD_SCRIPT_SIZE=$((UPLOAD_SCRIPT_SIZE/1024))
cat << EOF > "build/tahoe-upload_${VERSION}_amd64/DEBIAN/control"
Package: tahoe-upload
Version: $VERSION
Section: base
Priority: optional
Architecture: amd64
Maintainer: Robin Slot <robin@rkslot.nl>
Description: rsync-like upload program for Tahoe-LAFS
Installed-Size: $UPLOAD_SCRIPT_SIZE
Conflicts: tahoe-upload
Depends: python3 (>= 3.7), python3-requests, python3-tqdm
EOF

cp upload.py "build/tahoe-upload_${VERSION}_amd64/usr/bin/tahoe-upload"
chmod +rx "build/tahoe-upload_${VERSION}_amd64/usr/bin/tahoe-upload"

dpkg-deb --build --root-owner-group "build/tahoe-upload_${VERSION}_amd64"
