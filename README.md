# tahoe-pyfuse3

## Installation

### Native python

Install dependencies (debian):
```
apt install python3 python3-urllib3 python3-pyfuse3
```

Run `python3 mount.py`

### Docker

The image `derkades/tahoe-mount` and `derkades/tahoe-upload` is available on Docker Hub. Please note that mounting filesystems from docker containers is usually not a great idea, for production use please install packages on your host system instead.

### Debian
The package `tahoe-mount` is available in [my repository](https://deb.rkslot.nl). If you run into missing dependency issues on older Debian/Ubuntu versions, use the `tahoe-mount-static` version instead (amd64 only).

To build these packages locally, run `./build-deb.sh`.

## Features
- Listing files and directories
- Deleting (unlinking) files and directories
- Reading files (unlike tahoe's SFTP system, doesn't need to download the entire file)
- Creating directories
- Creating files
- Writing to files (broken due to tahoe bug, see below)

## Caveats
- Due to [severe file corruption bugs in Tahoe-LAFS](https://tahoe-lafs.org/trac/tahoe-lafs/ticket/3818) when writing to MDMF/SDMF mutable files, the file system won't allow writes (not even with `--read-only False`). To upload files, use the file upload script (see below).
- Mtime/ctime/crtime is not set properly. This will be addressed soon.

## Usage
```
mount.tahoe <root cap> <mountpoint> -o <options>
```

Options:
| name | value | description
| - | - | -
| `node_url` | string, standard url format | Tahoe-LAFS node web API URL (REQUIRED)
| `setuid` | int | User id which all files and directories in the filesystem should be owned by. Defaults to the user running the mount command (usually root).
| `setgid` | int | Group id, see above
| `file_mode` | int | Permission mode for files (default 644)
| `dir_mode` | int | Permission mode for directories (default 755)
| `ro` | none | Make filesystem read-only
| `allow_other` | none | Sets the FUSE allow_other option
| `debug` | none | Enables application debug logging
| `debug_fuse` | none | Enables FUSE debug logging
| `fork` | none | Fork before entering main filesystem loop, required for use in /etc/fstab
| `nofork` | none | Do not fork, see above (default)

Example:
```
mount.tahoe URI:DIR2:fz<snip>ge:lz<snip>ra /mnt/tahoe -o node_url=http://localhost:3456,file_mode=444,dir_mode=555,ro,nofork,allow_other
```
fstab:
```
URI:DIR2:fz<snip>ge:lz<snip>ra /mnt/tahoe tahoe node_url=http://localhost:3456,file_mode=444,dir_mode=555,ro,nofork,allow_other
```
