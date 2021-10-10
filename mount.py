#!/usr/bin/env python3
import os
from urllib.parse import quote
import requests

from argparse import ArgumentParser
import stat
import logging
import errno
import pyfuse3
import trio


try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


log = logging.getLogger(__name__)


class TestFs(pyfuse3.Operations):

    def __init__(self, node_url, root_cap):
        super(TestFs, self).__init__()
        self._node_url = node_url
        self._inode_to_cap = {pyfuse3.ROOT_INODE: ('dirnode', True, root_cap)}
        self._cap_to_inode = {root_cap: pyfuse3.ROOT_INODE}
        self._next_inode = pyfuse3.ROOT_INODE + 1
        self._open_handles = {}
        self.chunk_size = 5*128*1024  # nicely aligned with 5 tahoe 128KiB segments

    def _create_handle(self, data) -> int:
        for i in range(1000):
            if i not in self._open_handles:
                break

        self._open_handles[i] = data
        return i

    def _create_inode_from_json(self, child_json: list) -> int:
        (node_type, child) = child_json
        if 'rw_uri' in child:
            cap = child['rw_uri']
            mutable = True
        elif 'ro_uri' in child:
            cap = child['ro_uri']
            mutable = False
        else:
            raise ValueError(f'child does not contain rw_uri or ro_uri {child=}')

        return self._create_inode(node_type, mutable, cap)

    def _create_inode(self, node_type, mutable, cap) -> int:
        if cap in self._cap_to_inode:
            return self._cap_to_inode[cap]

        inode = self._next_inode
        self._next_inode = inode + 1
        self._inode_to_cap[inode] = (node_type, mutable, cap)
        self._cap_to_inode[cap] = inode
        return inode

    def _getattr(self, inode: int, node_type: str, size: int) -> pyfuse3.EntryAttributes:
        entry = pyfuse3.EntryAttributes()
        if node_type == 'dirnode':
            entry.st_mode = (stat.S_IFDIR | 0o755)
        elif node_type == 'filenode':
            entry.st_mode = (stat.S_IFREG | 0o644)
        else:
            raise ValueError(f'unknown type {node_type=}')

        entry.st_size = size

        stamp = int(1438467123.985654 * 1e9)
        entry.st_atime_ns = stamp
        entry.st_ctime_ns = stamp
        entry.st_mtime_ns = stamp
        entry.st_gid = os.getgid()
        entry.st_uid = os.getuid()
        entry.st_ino = inode

        return entry

    def getattr_json(self, inode, child_json, ctx=None) -> pyfuse3.EntryAttributes:
        (node_type, info) = child_json
        if node_type == 'filenode':
            size = int(info['ro_uri'].split(':')[-1])
        else:
            size = 0
        return self._getattr(inode, node_type, size)

    async def getattr(self, inode: int, ctx=None) -> pyfuse3.EntryAttributes:
        if inode not in self._inode_to_cap:
            raise ValueError(f'{inode=} unknown')

        (node_type, _mutable, cap) = self._inode_to_cap[inode]

        if node_type == 'filenode':
            size = int(cap.split(':')[-1])
        else:
            size = 0

        return self._getattr(inode, node_type, size)

    async def lookup(self, parent_inode: int, name: bytes, ctx=None) -> pyfuse3.EntryAttributes:
        if parent_inode not in self._inode_to_cap:
            raise ValueError(f'{parent_inode=} unknown')

        (node_type, _mutable, cap) = self._inode_to_cap[parent_inode]
        if node_type == 'filenode':
            raise(pyfuse3.FUSEError(errno.ENOTDIR))

        r_json = requests.get(self._node_url + '/uri/' + quote(cap), params={'t': 'json'}).json()

        assert r_json[0] == node_type

        name_str: str = name.decode()

        for child_name in r_json[1]['children'].keys():
            if child_name == name_str:
                child_json = r_json[1]['children'][child_name]
                inode = self._create_inode_from_json(child_json)
                return self.getattr_json(inode, child_json)

        raise pyfuse3.FUSEError(errno.ENOENT)

    async def opendir(self, inode, ctx):
        if inode not in self._inode_to_cap:
            raise pyfuse3.FUSEError(errno.ENOENT)

        (_typ, _mutable, cap) = self._inode_to_cap[inode]
        r_json = requests.get(self._node_url + '/uri/' + quote(cap), params={'t': 'json'}).json()
        nodetype = r_json[0]
        if nodetype == 'unknown':
            raise(pyfuse3.FUSEError(errno.ENOENT))
        elif nodetype == 'filenode':
            raise(pyfuse3.FUSEError(errno.ENOTDIR))
        elif nodetype != 'dirnode':
            raise(ValueError(f'{nodetype=}'))

        return self._create_handle(r_json)

    async def readdir(self, fh: int, start_id: int, token: pyfuse3.ReaddirToken):
        if fh not in self._open_handles:
            raise ValueError(f'file handle is not open? {fh=}')

        children: dict = self._open_handles[fh][1]['children']
        i = 0
        for child_name in children.keys():
            if i >= start_id:
                child = children[child_name]
                inode: int = self._create_inode_from_json(child)
                if not pyfuse3.readdir_reply(token, child_name.encode(), self.getattr_json(inode, child), i+1):
                    return
            i += 1

        return

    async def releasedir(self, fh: int):
        del self._open_handles[fh]

    async def mkdir(self, parent_inode: int, name: bytes, mode, ctx: pyfuse3.RequestContext) -> pyfuse3.EntryAttributes:
        if parent_inode not in self._inode_to_cap:
            raise ValueError(f'{parent_inode=} unknown')

        (parent_node_type, _parent_mutable, parent_cap) = self._inode_to_cap[parent_inode]
        if parent_node_type == 'filenode':
            raise(pyfuse3.FUSEError(errno.ENOTDIR))

        # Create directory
        cap = requests.post(self._node_url + '/uri', params={'t': 'mkdir', 'format': 'SDMF'}).text

        # Link directory
        requests.put(self._node_url + '/uri/' + quote(parent_cap) + '/' + quote(name.decode()),
                     params={'t': 'uri', 'replace': 'false'},
                     data=cap)

        inode = self._create_inode('dirnode', True, cap)
        return self._getattr(inode, 'dirnode', 0)

    async def open(self, inode, _flags, _ctx):
        # if flags & os.O_RDWR or flags & os.O_WRONLY:
        #     raise pyfuse3.FUSEError(errno.EACCES)
        (node_type, _mutable, cap) = self._inode_to_cap[inode]
        assert node_type == 'filenode'
        data = (cap, {})  # second element in tuple is for chunk cache
        fh = self._create_handle(data)
        log.debug('open fh %s', fh)
        return pyfuse3.FileInfo(fh=fh)

    async def read(self, fh, off, size):
        start_chunk = off // self.chunk_size
        end_chunk = (off + size) // self.chunk_size
        log.debug('read off=%s size=%s (to %s) start_chunk=%s end_chunk=%s', off, size, off+size, start_chunk, end_chunk)
        (cap, cache) = self._open_handles[fh]
        data = b''
        # in almost all cases, this for loop will only run for one iteration (usually chunk size is larger than read size)
        for chunk_index in range(start_chunk, end_chunk + 1):
            log.debug('chunk index %s', chunk_index)
            if chunk_index in cache:
                log.debug('cached')
                data += cache[chunk_index]
            else:
                log.debug('not cached')
                r_start = chunk_index * self.chunk_size
                r_end = r_start + self.chunk_size - 1  # tahoe doesn't care if this is beyond the end of the file
                chunk_data = requests.get(self._node_url + '/uri/' + quote(cap), headers={'Range': f'bytes={r_start}-{r_end}'}).content
                cache[chunk_index] = chunk_data
                data += chunk_data
        data_off = off % self.chunk_size
        return data[data_off:data_off+size]

    async def release(self, fh: int):
        log.debug('release fh %s', fh)
        del self._open_handles[fh]


def init_logging(debug=False):
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(threadName)s: '
                                  '[%(name)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    if debug:
        handler.setLevel(logging.DEBUG)
        root_logger.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def parse_args():
    '''Parse command line'''

    parser = ArgumentParser()

    parser.add_argument('mountpoint', type=str,
                        help='Where to mount the file system')
    parser.add_argument('node_url', type=str,
                        help='')
    parser.add_argument('root_cap', type=str,
                        help='')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debugging output')
    parser.add_argument('--debug-fuse', action='store_true', default=False,
                        help='Enable FUSE debugging output')
    return parser.parse_args()


def main():
    options = parse_args()
    init_logging(options.debug)

    testfs = TestFs(options.node_url, options.root_cap)
    fuse_options = set(pyfuse3.default_options)
    fuse_options.add('fsname=hello')
    fuse_options.add('allow_other')
    if options.debug_fuse:
        fuse_options.add('debug')
    pyfuse3.init(testfs, options.mountpoint, fuse_options)
    try:
        trio.run(pyfuse3.main)
    except:
        pyfuse3.close(unmount=False)
        raise

    pyfuse3.close()


if __name__ == '__main__':
    main()
