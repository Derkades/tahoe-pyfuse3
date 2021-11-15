#!/usr/bin/env python3
import os
import requests
import threading
import logging
import base64
from typing import Optional, List
from urllib.parse import quote

from argparse import ArgumentParser
import errno
import pyfuse3
import trio
import stat


try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


log = logging.getLogger(__name__)


class TahoeFs(pyfuse3.Operations):

    def __init__(self, node_url, root_cap, read_only):
        super(TahoeFs, self).__init__()
        self.supports_dot_lookup = False  # maybe it does?
        self.enable_writeback_cache = False
        self.enable_acl = False

        self._node_url = node_url
        self._inode_to_cap_dict = {pyfuse3.ROOT_INODE: root_cap}
        self._cap_to_inode_dict = {root_cap: pyfuse3.ROOT_INODE}
        self._next_inode = pyfuse3.ROOT_INODE + 1
        self._open_handles = {}
        self._inode_lock = threading.Lock()
        self._fh_lock = threading.Lock()
        self._chunk_size = 2*128*1024  # nicely aligned with tahoe 128KiB segments
        self.read_only = read_only

        try:
            self._find_cap_in_parent(pyfuse3.ROOT_INODE, None)
        except pyfuse3.FUSEError:
            pass

    def _create_handle(self, data) -> int:
        with self._fh_lock:
            for i in range(1000):
                if i not in self._open_handles:
                    self._open_handles[i] = data
                    return i

            raise Exception('out of handles')

    def _cap_to_inode(self, cap: str) -> int:
        with self._inode_lock:
            if cap in self._cap_to_inode_dict:
                return self._cap_to_inode_dict[cap]

            inode = self._next_inode
            self._next_inode = inode + 1
            self._inode_to_cap_dict[inode] = cap
            self._cap_to_inode_dict[cap] = inode
            return inode

    def _inode_to_cap(self, inode: int) -> Optional[str]:
        with self._inode_lock:
            if inode in self._inode_to_cap_dict:
                return self._inode_to_cap_dict[inode]
            else:
                return None

    def _cap_is_dir(self, cap: str) -> bool:
        return cap.split(':')[1] in {'DIR2', 'DIR2-MDMF'}

    def _cap_from_child_json(self, json: list) -> str:
        log.debug(json)
        json = json[1]
        return json['rw_uri'] if 'rw_uri' in json else json['ro_uri']

    def _find_cap_in_parent(self, parent_inode: int, name: str) -> str:
        cap = self._inode_to_cap(parent_inode)
        if not cap:
            raise ValueError(f'{parent_inode=} unknown')

        if not self._cap_is_dir(cap):
            raise(pyfuse3.FUSEError(errno.ENOTDIR))

        r_json = requests.get(self._node_url + '/uri/' + quote(cap), params={'t': 'json'}).json()

        for child_name in r_json[1]['children'].keys():
            if child_name == name:
                child = r_json[1]['children'][child_name]
                return self._cap_from_child_json(child)

        raise pyfuse3.FUSEError(errno.ENOENT)

    def _decode_lit(self, cap: str) -> bytes:
        content_b32 = cap.split(':')[2]
        actually_b32 = content_b32.upper() + '=' * ((8 - (len(content_b32) % 8)) % 8)
        return base64.b32decode(actually_b32)

    def _getattr(self, cap: str) -> pyfuse3.EntryAttributes:
        entry = pyfuse3.EntryAttributes()

        # TODO support LIT https://tahoe-lafs.readthedocs.io/en/latest/specifications/uri.html?highlight=URI%3ALIT#lit-uris

        split = cap.split(':')
        cap_type = split[1]
        if cap_type == 'DIR2':  # SDMF directory
            entry.st_mode = (stat.S_IFDIR | 0o755)
            entry.st_size = 0
        elif cap_type == 'DIR2-MDMF':  # MDMF directory
            raise NotImplementedError('MDMF directories not supported')
        elif cap_type == 'CHK':  # Immutable file
            entry.st_mode = (stat.S_IFREG | 0o444)
            entry.st_size = int(split[-1])
        elif cap_type == 'MDMF' or cap_type == 'MDMF-RO':  # MDMF file
            entry.st_mode = (stat.S_IFREG | 0o644)
            r_json = requests.get(self._node_url + '/uri/' + quote(cap), params={'t': 'json'}).json()
            entry.st_size = int(r_json[1]['size'])
        elif cap_type == 'SSK' or cap_type == 'SSK-RO':  # SDMF file
            raise NotImplementedError('SDMF files not supported')
        elif cap_type == 'LIT':
            entry.st_mode = (stat.S_IFREG | 0o444)
            entry.st_size = len(self._decode_lit(cap))
        else:
            raise NotImplementedError('cap not supported: ' + cap)

        stamp = int(1438467123.985654 * 1e9)
        entry.st_atime_ns = stamp
        entry.st_ctime_ns = stamp
        entry.st_mtime_ns = stamp
        entry.st_gid = os.getgid()
        entry.st_uid = os.getuid()
        entry.st_ino = self._cap_to_inode(cap)
        entry.st_blksize = 512
        entry.st_blocks = -(-entry.st_size // entry.st_blksize)  # ceil division

        return entry

    async def getattr(self, inode: int, ctx=None) -> pyfuse3.EntryAttributes:
        cap = self._inode_to_cap(inode)
        if not cap:
            raise ValueError(f'{inode=} unknown')

        return self._getattr(cap)

    async def lookup(self, parent_inode: int, name: bytes, _ctx=None) -> pyfuse3.EntryAttributes:
        cap: str = self._find_cap_in_parent(parent_inode, name.decode())
        return self._getattr(cap)

    async def opendir(self, inode, _ctx):
        cap = self._inode_to_cap(inode)
        if not cap:
            raise ValueError(f'{inode=} unknown')

        r_json = requests.get(self._node_url + '/uri/' + quote(cap), params={'t': 'json'}).json()

        return self._create_handle(r_json)

    async def readdir(self, fh: int, start_id: int, token: pyfuse3.ReaddirToken):
        if fh not in self._open_handles:
            raise ValueError(f'file handle is not open? {fh=}')

        children: dict = self._open_handles[fh][1]['children']

        i = 0
        for child_name in children.keys():
            if i >= start_id:
                child = children[child_name]
                cap = self._cap_from_child_json(child)
                if not pyfuse3.readdir_reply(token, child_name.encode(), self._getattr(cap), i+1):
                    return
            i += 1

        return

    async def releasedir(self, fh: int):
        del self._open_handles[fh]

    async def mkdir(self, parent_inode: int, name: bytes, _mode, _ctx: pyfuse3.RequestContext) -> pyfuse3.EntryAttributes:
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        parent_cap = self._inode_to_cap(parent_inode)
        if not parent_cap:
            raise ValueError(f'{parent_inode=} unknown')

        if not self._cap_is_dir(parent_cap):
            raise(pyfuse3.FUSEError(errno.ENOTDIR))

        # Create directory
        cap = requests.post(self._node_url + '/uri', params={'t': 'mkdir', 'format': 'SDMF'}).text

        # Link directory
        requests.put(self._node_url + '/uri/' + quote(parent_cap) + '/' + quote(name.decode()),
                     params={'t': 'uri', 'replace': 'false'},
                     data=cap)

        return self._getattr(cap)

    async def create(self, parent_inode, name, _mode, flags, ctx) -> pyfuse3.EntryAttributes:
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        log.debug('create')
        if mode & stat.S_IFREG == stat.S_IFREG:
            parent_cap = self._inode_to_cap(parent_inode)
            if not parent_cap:
                raise ValueError(f'{parent_inode=} unknown')

            if not self._cap_is_dir(parent_cap):
                raise(pyfuse3.FUSEError(errno.ENOTDIR))

            # Create file
            cap = requests.put(self._node_url + '/uri/' + quote(parent_cap) + '/' + quote(name.decode()),
                               params={'format': 'MDMF'}).text

            inode = self._cap_to_inode(cap)
            fi = self.open(inode, flags, ctx)
            return fi, self._getattr(cap)
        else:
            raise NotImplementedError('unsupported mode: ' + mode)

    async def move(self, old_inode_p: int, old_name: str, new_inode_p: int, new_name: str, flags, ctx: pyfuse3.RequestContext):
        if flags & pyfuse3.RENAME_EXCHANGE == pyfuse3.RENAME_EXCHANGE:
            raise pyfuse3.FUSEError(errno.ENOTSUP)

        if flags & pyfuse3.RENAME_NOREPLACE == pyfuse3.RENAME_NOREPLACE:
            replace_mode = 'false'
        else:
            replace_mode = 'only-files'

        old_pcap = self._inode_to_cap(old_inode_p)
        new_pcap = self._inode_to_cap(new_inode_p)
        assert old_pcap
        assert new_pcap

        r = requests.post(self._node_url + '/uri/' + quote(old_pcap) + '/?=relink'
                          '&from_name=' + quote(old_name) + \
                          '&to_dir=' + quote(new_pcap) + \
                          '&to_name=' + quote(new_name) + \
                          '&replace=' + replace_mode)

        if r.status_code == 409:
            raise pyfuse3.FUSEError(errno.EEXIST)

        if r.status_code != 200:
            raise Exception('Unexpected response code ' + r.status_code)

    async def open(self, inode, flags, _ctx):
        if flags & os.O_TRUNC == os.O_TRUNC:
            if not self.read_only:
                raise Exception('Truncate not supported')

        cap = self._inode_to_cap(inode)
        assert cap
        data = (cap, {})  # second element in tuple is for chunk cache
        fh = self._create_handle(data)
        log.debug('open fh %s', fh)
        return pyfuse3.FileInfo(fh=fh)

    def _download_range(self, cap: str, start: int, end_excl: int) -> bytes:
        r = requests.get(self._node_url + '/uri/' + quote(cap),
                         headers={
                             'Range': f'bytes={start}-{end_excl - 1}'
                         })
        if r.status_code == 416:
            log.warning('cap %s was read beyond the end of the file at offset %s', cap, start)
            return b''

        if r.status_code not in {200, 206}:
            raise Exception('unexpected status code ' + str(r.status_code))

        return r.content

    def _cache_chunks(self, cap: str, cache, chunks_to_download: List[int]) -> None:
        if len(chunks_to_download) == 0:
            return

        ranges_to_download = []

        lowest_index = chunks_to_download[0]
        highest_index = lowest_index
        for chunk_index in chunks_to_download[1:]:
            if chunk_index == highest_index + 1:
                highest_index = chunk_index
            else:
                ranges_to_download.append((lowest_index, highest_index))
                lowest_index = chunk_index
                highest_index = chunk_index
        ranges_to_download.append((lowest_index, highest_index))

        log.debug('downloading ranges %s', ranges_to_download)

        for ranges_to_download in ranges_to_download:
            c_start, c_end_incl = ranges_to_download
            # make one large request for better throughput
            r_start = c_start * self._chunk_size
            r_end_excl = (c_end_incl+1) * self._chunk_size
            log.debug('downloading range %s-%s (incl) bytes %s-%s (excl) size %skiB',
                      c_start, c_end_incl, r_start, r_end_excl, (r_end_excl-r_start) // 1024)
            data = self._download_range(cap, r_start, r_end_excl)
            log.debug('size of returned data: %s', len(data))
            # now, split up this large block back into chunks
            for chunk_index in range(c_start, c_end_incl + 1):
                local_start = (chunk_index-c_start) * self._chunk_size
                local_end = (chunk_index-c_start+1) * self._chunk_size
                chunk_data = data[local_start:local_end]
                log.debug('storing chunk index %s in chunk cache, size %s (from %s to %s excl)', chunk_index, len(chunk_data), local_start, local_end)
                cache[chunk_index] = chunk_data

    async def read(self, fh: int, off: int, size: int) -> bytes:
        (cap, cache) = self._open_handles[fh]

        cap_type = cap.split(':')[1]

        if cap_type == 'LIT':
            data = self._decode_lit(cap)
            return data[off:off+size]
        elif cap_type in {'CHK', 'MDMF', 'MDMF-RO', 'SSK', 'SSK-RO'}:
            log.debug('read off=%skiB, size=%skiB', off // 1024, size // 1024)
            if size >= 65536:
                prefetch_blocks = 2
                if size >= 131072:
                    prefetch_blocks = 5
                if size >= 262144:
                    prefetch_blocks = 15

                # chunks we actually need to read
                r_start_chunk = off // self._chunk_size
                r_end_chunk = (off + size) // self._chunk_size

                # chunks we want to cache
                c_start_chunk = r_start_chunk
                prefetch_count = prefetch_blocks - r_end_chunk % prefetch_blocks
                c_end_chunk = r_end_chunk + prefetch_count
                log.debug('use chunk cache with start=%s, end=%s, chunks=%s, prefetch_blocks=%s, prefetch_count=%s, total_size=%skiB',
                        c_start_chunk,
                        c_end_chunk,
                        c_end_chunk - c_start_chunk + 1,
                        prefetch_blocks,
                        prefetch_count,
                        ((c_end_chunk - c_start_chunk + 1) * self._chunk_size) // 1024)

                c_chunks = range(c_start_chunk, c_end_chunk+1)
                chunks_to_download = [i for i in c_chunks if i not in cache]
                log.debug('all chunks %s, chunks to download: %s', list(c_chunks), chunks_to_download)
                self._cache_chunks(cap, cache, chunks_to_download)

                # now that all data has been downloaded and cached, return the data we were asked for

                data = b''
                for chunk_index in range(r_start_chunk, r_end_chunk+1):
                    log.debug('getting chunk %s from cache', chunk_index)
                    data += cache[chunk_index]
                data_off = off % self._chunk_size
                return data[data_off:data_off+size]
            else:
                log.debug('don\'t use chunk cache')
                return self._download_range(cap, off, off+size)
        else:
            raise NotImplementedError("cap not supported: " + cap)

    async def write(self, fh: int, off: int, buf: bytes) -> int:
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        raise pyfuse3.FUSEError(errno.ENOTSUP)

        start_chunk = off // self._chunk_size
        end_chunk = (off + len(buf)) // self._chunk_size
        log.debug('write off=%s size=%s (to %s)', off, len(buf), off + len(buf))
        (cap, cache) = self._open_handles[fh]
        for chunk_index in range(start_chunk, end_chunk + 1):
            # ideally we should update the local cache instead of destroying it
            if chunk_index in cache:
                del cache[chunk_index]

        params = {'offset': off}
        r = requests.put(self._node_url + '/uri/' + quote(cap), params=params, data=buf)
        if r.status_code != 200:
            log.warning(r.text)
        return len(buf)

    async def release(self, fh: int):
        log.debug('release fh %s', fh)
        del self._open_handles[fh]

    async def unlink(self, parent_inode: int, name: bytes, ctx: pyfuse3.RequestContext()):
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        cap = self._inode_to_cap(parent_inode)
        r = requests.delete(f'{self._node_url}/uri/{quote(cap)}/{quote(name.decode())}')
        assert r.status_code == 200


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
    """Parse command line"""

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
    parser.add_argument('--read-only', default=False,
                        help='Don\'t allow writing to, modifying, deleting or creating files or directories.')
    return parser.parse_args()


def main():
    options = parse_args()
    init_logging(options.debug)

    testfs = TahoeFs(options.node_url, options.root_cap, options.read_only)
    fuse_options = set(pyfuse3.default_options)
    fuse_options.add('fsname=tahoe')
    fuse_options.add('allow_other')
    if options.debug_fuse:
        fuse_options.add('debug')
    pyfuse3.init(testfs, options.mountpoint, fuse_options)
    try:
        trio.run(pyfuse3.main)
    except:
        pyfuse3.close(unmount=True)
        raise

    pyfuse3.close()


if __name__ == '__main__':
    main()
