# tahoe-pyfuse3

## Installation

### Native python

Install dependencies (debian):
```
apt install python3 python3-urllib3 python3-pyfuse3 python3-tqdm
```

Run `python3 mount.py` or `python3 upload.py`

### Docker

Images `derkades/tahoe-mount` and `derkades/tahoe-upload` are available on Docker Hub. You can also build them locally using `./build-docker.sh`

### Static executables
Run `./build-static.sh` (uses Docker so it doesn't litter your system with crap) and you'll find executables in `./build`. Copy `mount.tahoe` to `/usr/local/sbin` for usage in `/etc/fstab`.

### Debian
Packages `tahoe-mount` and `tahoe-upload` are available for amd64 in [my repository](https://deb.rkslot.nl).

## Mount client

### Features
- Listing files and directories
- Deleting (unlinking) files and directories
- Reading files (unlike tahoe's SFTP system, doesn't need to download the entire file)
- Creating directories
- Creating files
- Writing to files (broken due to tahoe bug, see below)

### Caveats
- Due to [severe file corruption bugs in Tahoe-LAFS](https://tahoe-lafs.org/trac/tahoe-lafs/ticket/3818) when writing to MDMF/SDMF mutable files, the file system won't allow writes (not even with `--read-only False`). To upload files, use the file upload script (see below).
- Mtime/ctime/crtime is not set properly. This will be addressed soon.

### Usage
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

## File uploader
Like `rsync`, the `upload.py` script can upload local files and directories to a Tahoe-LAFS directory.


### Behavior
The upload script will recursively create directories and upload files in Tahoe-LAFS. Its duplicate file/directory behavior is best described using pseudocode:

```
if local is file:
    if remote is file:
        if remote and local file are same size:
            don't upload
        else:
            delete and re-upload
    else if remote is directory:
        delete directory and upload file
    else if remote doesn't exist:
        upload file
else if directory:
    if remote is file:
        delete file and create directory
    else if remote is directory:
        do nothing
    else if remote doesn't exist:
        create directory

    repeat for all files in this directory
```
The upload script will never move/delete/create/modify local files/directories.

### Usage
```
usage: upload.py [-h] path api cap

positional arguments:
  path        Path to file or directory to upload. Like rsync, add a trailing slash to upload directory contents, no trailing slash to upload the directory itself.
  api         HTTP REST API URL of a Tahoe-LAFS node
  cap         Tahoe directory capability where files should be uploaded to

optional arguments:
  -h, --help  show this help message and exit
```

Example:
```
python3 upload.py /some/dir http://localhost:3456 URI:DIR2:fzwyukltbehjx37nuyp6wy2qge:lzzg3oy2okmfcblquvoyp7qtq6xge2ptge6srogn56hbn7ckhgra
```
