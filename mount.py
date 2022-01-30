#!/usr/bin/env python3
import argparse
import base64
import errno
import json
import logging
import logging.handlers
import os
import stat
import threading
import time
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, cast
from urllib.parse import quote, urlparse

import _pyfuse3  # for pyinstaller pylint: disable=unused-import
import pyfuse3
import trio
import urllib3
from pyfuse3 import FUSEError
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.connectionpool import ConnectionPool
from urllib3.exceptions import HTTPError

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


log = logging.getLogger(__name__)


class HandleData():
    pass


class FileHandleData(HandleData):
    def __init__(self, cap: str, cache: Dict[int, bytes]):
        self.cap = cap
        self.cache = cache
        self.read_count = 0


class DirHandleData(HandleData):
    def __init__(self, children: Dict[str, Any]):
        self.children = children


class TahoeFs(pyfuse3.Operations):

    def __init__(self, node_urls: List[str], root_cap: str, read_only: bool,
                 uid: int, gid: int, dir_mode: int, file_mode: int) -> None:
        super(TahoeFs, self).__init__()
        self.supports_dot_lookup = False  # Tahoe-LAFS supports directory symlinks, we can't know a directory's parent
        self.enable_writeback_cache = False
        self.enable_acl = False

        self._read_only = read_only
        self._uid = uid
        self._gid = gid
        self._dir_mode = dir_mode
        self._file_mode = file_mode

        self._inode_to_cap_dict = {pyfuse3.ROOT_INODE: root_cap}
        self._cap_to_inode_dict = {root_cap: pyfuse3.ROOT_INODE}
        self._next_inode = pyfuse3.ROOT_INODE + 1
        self._open_handles: Dict[int, HandleData] = {}
        self._inode_lock = threading.Lock()
        self._fh_lock = threading.Lock()

        self._chunk_size = 2*128*1024  # nicely aligned with tahoe 128 KiB segments
        self._max_cached_chunks = 64*1024*1024 // self._chunk_size  # 64 MiB

        self._common_headers = {
            'User-Agent': 'tahoe-mount',
            'Accept': 'text/plain'
        }

        self._connection_pools: List[ConnectionPool] = []
        self._next_connection_pool_index = 0
        retry_config = urllib3.Retry(total=2, connect=2, read=2, redirect=0, other=0)
        timeout_config = urllib3.Timeout(total=10)
        for node_url in node_urls:
            parsed_url = urlparse(node_url)
            pool_kwargs = {
                'host': parsed_url.hostname,
                'headers': self._common_headers,
                'retries': retry_config,
                'timeout': timeout_config,
                'maxsize': 2
            }

            if parsed_url.scheme == 'http':
                pool_kwargs['port'] = parsed_url.port if parsed_url.port is not None else 80
                pool = HTTPConnectionPool(**pool_kwargs)
            elif parsed_url.scheme == 'https':
                pool_kwargs['port'] = parsed_url.port if parsed_url.port is not None else 443
                pool = HTTPSConnectionPool(**pool_kwargs)
            self._connection_pools.append(pool)

    def _get_connection_pool(self) -> urllib3.connectionpool.ConnectionPool:
        pool = self._connection_pools[self._next_connection_pool_index]
        self._next_connection_pool_index += 1
        if self._next_connection_pool_index >= len(self._connection_pools):
            self._next_connection_pool_index = 0
        return pool

    def _create_handle(self, data: HandleData) -> int:
        with self._fh_lock:
            for fh in range(1000):
                if fh not in self._open_handles:
                    self._open_handles[fh] = data
                    return fh

            raise FUSEError(errno.EMFILE)

    def _get_handle(self, fh: int) -> Optional[HandleData]:
        with self._fh_lock:
            if fh in self._open_handles:
                return self._open_handles[fh]
            else:
                return None

    def _del_hande(self, fh: int) -> None:
        with self._fh_lock:
            del self._open_handles[fh]

    def _cap_to_inode(self, cap: str) -> int:
        with self._inode_lock:
            if cap in self._cap_to_inode_dict:
                return self._cap_to_inode_dict[cap]

            self._inode_to_cap_dict[self._next_inode] = cap
            self._cap_to_inode_dict[cap] = self._next_inode
            self._next_inode += 1
            return self._next_inode - 1

    def _inode_to_cap(self, inode: int) -> Optional[str]:
        with self._inode_lock:
            if inode in self._inode_to_cap_dict:
                return self._inode_to_cap_dict[inode]
            else:
                return None

    def _cap_is_dir(self, cap: str) -> bool:
        return cap.split(':')[1] in {'DIR2', 'DIR2-RO', 'DIR2-MDMF'}

    def _cap_from_child_json(self, child_json: list) -> str:
        json2: Dict[str, Any] = child_json[1]
        return json2['rw_uri'] if 'rw_uri' in json2 else json2['ro_uri']

    def _find_cap_in_parent(self, parent_inode: int, name: str) -> str:
        """
        Look up the file cap corresponding to a name in a parent directory.
        Parameters
            parent_inode: Inode of the parent directory. Raises FUSEError(errno.ENOTDIR) if it's not a directory.
            name: Name of the file/directory to look for.
        Returns
            Capability. Raises FUSEError(errno.ENOENT) if the file was not found.
        """
        cap = self._inode_to_cap(parent_inode)
        if not cap:
            raise ValueError(f'parent_inode={parent_inode} unknown')

        if not self._cap_is_dir(cap):
            log.warning('_find_cap_in_parent() was called with parent_inode=%s name=%s but '
                        'this inode is not a directory', parent_inode, name)
            raise FUSEError(errno.ENOTDIR)

        try:
            resp = self._get_connection_pool().request('GET',
                                                       '/uri/' + quote(cap) + '?t=json')
        except HTTPError as e:
            log.warning('error during GET request for _find_cap_in_parent(): %s', e)
            raise FUSEError(errno.EREMOTEIO) from e

        if resp.status != 200:
            log.warning('unexpected status code %s _find_cap_in_parent() GET request', resp.status)
            raise FUSEError(errno.EREMOTEIO)

        r_json = json.loads(resp.data.decode())

        for child_name in r_json[1]['children'].keys():
            if child_name == name:
                child = r_json[1]['children'][child_name]
                return self._cap_from_child_json(child)

        raise FUSEError(errno.ENOENT)

    def _decode_lit(self, cap: str) -> bytes:
        content_b32 = cap.split(':')[2]
        actually_b32 = content_b32.upper() + '=' * ((8 - (len(content_b32) % 8)) % 8)
        return base64.b32decode(actually_b32)

    def _getattr(self, cap: str) -> pyfuse3.EntryAttributes:
        entry = pyfuse3.EntryAttributes()

        split = cap.split(':')
        cap_type = split[1]
        if cap_type == 'DIR2' or cap_type == 'DIR2-RO':  # SDMF directory
            entry.st_mode = (stat.S_IFDIR | self._dir_mode)
            entry.st_size = 0
        elif cap_type == 'DIR2-MDMF':  # MDMF directory
            raise NotImplementedError('MDMF directories not supported')
        elif cap_type == 'CHK':  # Immutable file
            entry.st_mode = (stat.S_IFREG | self._file_mode)
            entry.st_size = int(split[-1])
        elif cap_type == 'MDMF' or cap_type == 'MDMF-RO':  # MDMF file
            entry.st_mode = (stat.S_IFREG | self._file_mode)

            try:
                resp = self._get_connection_pool().request('GET',
                                                           '/uri/' + quote(cap) + '?t=json')
            except HTTPError as e:
                log.warning('error during GET request for _getattr(): %s', e)
                raise FUSEError(errno.EREMOTEIO) from e

            if resp.status != 200:
                log.warning('unexpected status code %s _getattr() MDMF(-RO) GET request', resp.status)
                raise FUSEError(errno.EREMOTEIO)

            r_json = json.loads(resp.data.decode())
            entry.st_size = int(r_json[1]['size'])
        elif cap_type == 'SSK' or cap_type == 'SSK-RO':  # SDMF file
            log.error('SDMF files not supported')
            raise FUSEError(errno.EREMOTEIO)
        elif cap_type == 'LIT':
            entry.st_mode = (stat.S_IFREG | self._file_mode)
            entry.st_size = len(self._decode_lit(cap))
        else:
            log.error('cap not supported: %s', cap)
            raise FUSEError(errno.EREMOTEIO)

        stamp = int(1438467123.985654 * 1e9)
        entry.st_atime_ns = stamp
        entry.st_ctime_ns = stamp
        entry.st_mtime_ns = stamp
        entry.st_uid = self._uid
        entry.st_gid = self._gid
        entry.st_ino = self._cap_to_inode(cap)
        entry.st_blksize = 512
        entry.st_blocks = -(-entry.st_size // entry.st_blksize)  # ceil division

        return entry

    async def getattr(self, inode: int, _ctx: pyfuse3.RequestContext) -> pyfuse3.EntryAttributes:
        cap = self._inode_to_cap(inode)
        if not cap:
            raise ValueError(f'inode {inode} unknown')

        return self._getattr(cap)

    async def lookup(self, parent_inode: int, name: bytes, _ctx: pyfuse3.RequestContext) -> pyfuse3.EntryAttributes:
        log.debug('lookup parent_inode=%s name=%s', parent_inode, name)
        cap: str = self._find_cap_in_parent(parent_inode, name.decode())
        return self._getattr(cap)

    async def opendir(self, inode: int, _ctx: pyfuse3.RequestContext) -> int:
        cap = self._inode_to_cap(inode)
        if not cap:
            raise ValueError(f'inode {inode} unknown')

        try:
            resp = self._get_connection_pool().request('GET',
                                                       '/uri/' + quote(cap) + '?t=json')
        except HTTPError as e:
            log.warning('error during GET request for opendir(): %s', e)
            raise FUSEError(errno.EREMOTEIO) from e

        if resp.status != 200:
            log.warning('unexpected status code %s opendir() GET request: %s', resp.status, resp.data)
            raise FUSEError(errno.EREMOTEIO)

        r_json = json.loads(resp.data.decode())
        children: Dict[str, Any] = r_json[1]['children']
        return self._create_handle(DirHandleData(children))

    async def readdir(self, fh: int, start_id: int, token: pyfuse3.ReaddirToken) -> None:
        handle_data = cast(Optional[DirHandleData], self._get_handle(fh))

        if handle_data is None:
            raise ValueError(f'file handle is not open? {fh}')

        children: Dict[str, Any] = handle_data.children

        i = 0
        for child_name in children.keys():
            if i >= start_id:
                child_json: List[Any] = children[child_name]
                cap: str = self._cap_from_child_json(child_json)
                if not pyfuse3.readdir_reply(token, child_name.encode(), self._getattr(cap), i+1):
                    return
            i += 1

        return

    async def releasedir(self, fh: int) -> None:
        self._del_hande(fh)

    async def mkdir(self, parent_inode: int, name: bytes,
                    _mode: int, _ctx: pyfuse3.RequestContext) -> pyfuse3.EntryAttributes:
        if self._read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        parent_cap = self._inode_to_cap(parent_inode)
        if not parent_cap:
            raise ValueError(f'parent_inode {parent_inode} unknown')

        if not self._cap_is_dir(parent_cap):
            raise pyfuse3.FUSEError(errno.ENOTDIR)

        # Create directory
        try:
            resp = self._get_connection_pool().request('POST',
                                                       '/uri?t=mkdir&format=SDMF')
        except HTTPError as e:
            log.warning('error during POST request for mkdir(): %s', e)
            raise FUSEError(errno.EREMOTEIO) from e

        if resp.status != 201:
            log.warning('unexpected status code %s opendir() POST request: %s', resp.status, resp.data)
            raise FUSEError(errno.EREMOTEIO)
        cap = resp.data.decode()

        # Link directory
        try:
            resp = self._get_connection_pool().request('PUT',
                                                    '/uri/' + quote(parent_cap) + '/' + quote(name.decode()),
                                                    body=cap)
        except HTTPError as e:
            log.warning('error during PUT request for mkdir(): %s', e)
            raise FUSEError(errno.EREMOTEIO) from e

        if resp.status != 201:
            log.warning('unexpected status code %s opendir() GET request: %s', resp.status, resp.data)
            raise FUSEError(errno.EREMOTEIO)

        return self._getattr(cap)

    async def create(self, parent_inode: int, name: bytes, mode: int,
                     flags: int, ctx: pyfuse3.RequestContext) -> pyfuse3.EntryAttributes:
        if self._read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        log.debug('create')
        if mode & stat.S_IFREG == stat.S_IFREG:
            parent_cap = self._inode_to_cap(parent_inode)
            if not parent_cap:
                raise ValueError(f'parent_inode {parent_inode} unknown')

            if not self._cap_is_dir(parent_cap):
                raise pyfuse3.FUSEError(errno.ENOTDIR)

            try:
                url = '/uri/' + quote(parent_cap) + '/' + quote(name.decode()) + "?format=MDMF"
                resp = self._get_connection_pool().request('PUT', url)
            except HTTPError as e:
                log.warning('error during PUT request for create(): %s', e)
                raise FUSEError(errno.EREMOTEIO) from e

            if resp.status != 201:
                log.warning('unexpected status code %s create() PUT request', resp.status)
                raise FUSEError(errno.EREMOTEIO)

            cap = resp.data.decode()
            inode = self._cap_to_inode(cap)
            file_info = self.open(inode, flags, ctx)
            return file_info, self._getattr(cap)
        else:
            raise NotImplementedError('unsupported mode: ' + oct(mode))

    async def rename(self, parent_inode_old: int, name_old: str, parent_inode_new: int, name_new: str,
                     flags: int, _ctx: pyfuse3.RequestContext) -> None:
        if flags & pyfuse3.RENAME_EXCHANGE == pyfuse3.RENAME_EXCHANGE:
            raise pyfuse3.FUSEError(errno.ENOTSUP)

        if flags & pyfuse3.RENAME_NOREPLACE == pyfuse3.RENAME_NOREPLACE:
            replace_mode = 'false'
        else:
            replace_mode = 'only-files'

        old_pcap = self._inode_to_cap(parent_inode_old)
        new_pcap = self._inode_to_cap(parent_inode_new)
        assert old_pcap
        assert new_pcap

        try:
            resp = self._get_connection_pool().request('POST',
                                                       '/uri' + quote(old_pcap) + '/?=relink'
                                                       '&from_name=' + quote(name_old) +
                                                       '&to_dir=' + quote(new_pcap) +
                                                       '&to_name=' + quote(name_new) +
                                                       '&replace=' + replace_mode)
        except HTTPError as e:
            log.warning('error during POST request for move(): %s', e)
            raise FUSEError(errno.EREMOTEIO) from e

        if resp.status == 409:
            raise pyfuse3.FUSEError(errno.EEXIST)

        if resp.status != 200:
            log.warning('unexpected status code %s move() POST request', resp.status)
            raise FUSEError(errno.EREMOTEIO)

    async def open(self, inode: int, flags: int, _ctx: pyfuse3.RequestContext) -> pyfuse3.FileInfo:
        if flags & os.O_TRUNC == os.O_TRUNC:
            if not self._read_only:
                raise Exception('Truncate not supported')

        cap = self._inode_to_cap(inode)
        assert cap
        fh = self._create_handle(FileHandleData(cap, cache={}))
        log.debug('open fh %s', fh)
        return pyfuse3.FileInfo(fh=fh)

    def _download_range(self, cap: str, start: int, end_excl: int) -> bytes:
        try:
            resp = self._get_connection_pool().request('GET',
                                                       '/uri/' + quote(cap),
                                                       headers={
                                                           **self._common_headers,
                                                           'Range': f'bytes={start}-{end_excl - 1}'
                                                       })
        except HTTPError as e:
            log.warning('error during GET request for _download_range(): %s', e)
            raise FUSEError(errno.EREMOTEIO) from e

        if resp.status == 416:
            log.warning('cap %s was read beyond the end of the file at offset %s', cap, start)
            return b''

        if resp.status not in {200, 206}:
            log.warning('unexpected status code %s _download_range() GET request', resp.status)
            raise FUSEError(errno.EREMOTEIO)

        log.debug('done')

        return resp.data

    async def _download_chunk_range(self, cap, c_start, c_end_incl) -> bytes:
        start_time = time.perf_counter()

        r_start = c_start * self._chunk_size
        r_end = (c_end_incl+1) * self._chunk_size
        log.debug('downloading %s chunks %s-%s bytes %s-%s (size: %s KiB)',
                  c_end_incl - c_start + 1, c_start, c_end_incl, r_start, r_end, (r_end - r_start) // 1024)

        if c_end_incl - c_start + 1 >= 8:
            log.debug('parallel download')
            r_middle = r_start + (r_end - r_start) // 2
            with ThreadPool(2) as pool:
                args = (
                    (cap, r_start, r_middle + 1),
                    (cap, r_middle, r_end)
                )
                datas = pool.starmap(self._download_range, args)
                data = b''.join(datas)
        else:
            data = self._download_range(cap, r_start, r_end)

        duration = time.perf_counter() - start_time
        throughput = len(data) / (duration * 1024 * 1024)
        log.debug('downloaded in %.1fs at %.1f Mb/s or %.1f MB/s', duration, throughput*8, throughput)
        return data

    def _prune_cache(self, c_start: int, cache: Dict[int, bytes]):
        """
        Deletes chunks from cache if it's too large
        """
        if len(cache) < self._max_cached_chunks:
            return

        log.debug('chunk cache has surpassed max size, clearing cached chunks')
        # chunks before our current chunk are likely to never be read again, try to clear those first
        to_clear = []
        for chunk_index in cache.keys():
            if chunk_index >= c_start:
                break
            to_clear.append(chunk_index)

        for chunk_index in to_clear:
            del cache[chunk_index]

        if len(cache) > self._max_cached_chunks:
            log.debug("cache is still to large after first pass, clearing entirely")
            cache.clear()

    async def read(self, fh: int, off: int, size: int) -> bytes:
        handle_data = cast(Optional[FileHandleData], self._get_handle(fh))
        if handle_data is None:
            return FUSEError(errno.ENOENT)  # TODO What is the appropriate errno for invalid file handle?

        cap = handle_data.cap
        cache = handle_data.cache

        cap_type = cap.split(':')[1]

        if cap_type == 'LIT':
            data = self._decode_lit(cap)
            return data[off:off+size]
        elif cap_type in {'CHK', 'MDMF', 'MDMF-RO', 'SSK', 'SSK-RO'}:
            log.debug('read off=%skiB, size=%skiB', off // 1024, size // 1024)
            if size >= 65536:
                if size >= 131072:
                    prefetch_chunk_count = 2
                else:
                    prefetch_chunk_count = 1

                if handle_data.read_count > 3:
                    prefetch_chunk_count *= 8
                elif handle_data.read_count > 1:
                    prefetch_chunk_count *= 2

                handle_data.read_count += 1

                c_start = off // self._chunk_size
                c_end_incl = -((off + size) // -self._chunk_size)
                c_end_incl += prefetch_chunk_count - c_end_incl % prefetch_chunk_count

                c_start_download = c_start
                while c_start_download in cache:
                    # this chunk is cached, we can start downloading from the next chunk
                    c_start_download += 1

                cache_data = b''
                for chunk_index in range(c_start, c_start_download):  # add our cache if we have any
                    cache_data += cache[chunk_index]

                if c_start_download <= c_end_incl:
                    download_data = await self._download_chunk_range(cap, c_start_download, c_end_incl)
                else:
                    download_data = b''

                data = cache_data + download_data

                # split received data into chunks and store in cache
                for chunk_index in range(c_start, c_end_incl + 1):
                    local_start = (chunk_index-c_start) * self._chunk_size
                    local_end = (chunk_index-c_start+1) * self._chunk_size
                    chunk_data = data[local_start:local_end]
                    cache[chunk_index] = chunk_data

                # prune cache, we can safely do this now since we won't read from cache anymore (until the next read)
                self._prune_cache(c_start, cache)

                # get the data we actually want to read
                data_off = off % self._chunk_size
                return data[data_off:data_off+size]
            else:
                log.debug("don't use chunk cache")
                return self._download_range(cap, off, off+size)
        else:
            raise NotImplementedError("cap not supported: " + cap)

    async def write(self, _fh: int, _off: int, _buf: bytes) -> int:
        if self._read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        raise pyfuse3.FUSEError(errno.ENOTSUP)

        # start_chunk = off // self._chunk_size
        # end_chunk = (off + len(buf)) // self._chunk_size
        # log.debug('write off=%s size=%s (to %s)', off, len(buf), off + len(buf))

        # data = self._get_handle(fh)
        # cap: str = data.cap
        # cache: Dict[int, bytes] = data.cache

        # for chunk_index in range(start_chunk, end_chunk + 1):
        #     # ideally we should update the local cache instead of destroying it
        #     if chunk_index in cache:
        #         del cache[chunk_index]

        # try:
        #     resp = self._get_connection_pool().request('PUT',
        #                                                '/uri/' + quote(cap) + '?offset=' + str(off),
        #                                                body=buf)
        # except HTTPError as e:
        #     log.warning('error during PUT request for write(): %s', e)
        #     raise FUSEError(errno.EREMOTEIO) from e

        # if resp.status != 200:
        #     log.warning('unexpected status code %s write() PUT request', resp.status)
        #     raise FUSEError(errno.EREMOTEIO)

        # return len(buf)

    async def release(self, fh: int) -> None:
        log.debug('release fh %s', fh)
        self._del_hande(fh)

    async def unlink(self, parent_inode: int, name: bytes, _ctx: pyfuse3.RequestContext) -> None:
        if self._read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        pcap = self._inode_to_cap(parent_inode)
        assert pcap is not None

        try:
            resp = self._get_connection_pool().request('DELETE',
                                                       '/uri/' + quote(pcap) + '/' + quote(name.decode()))
        except HTTPError as e:
            log.warning('error during DELETE request for unlink(): %s', e)
            raise FUSEError(errno.EREMOTEIO) from e

        if resp.status != 200:
            log.warning('unexpected status code %s unlink() DELETE request', resp.status)
            raise FUSEError(errno.EREMOTEIO)


def init_logging(syslog: bool, debug: bool = False) -> None:
    if syslog:
        handler = logging.handlers.SysLogHandler(address='/dev/log')
        formatter = logging.Formatter('tahoe-mount: [%(threadName)s] [%(name)s] %(message)s')
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(threadName)s] [%(name)s] %(message)s',
                                      datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    if debug:
        handler.setLevel(logging.DEBUG)
        root_logger.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def parse_args() -> argparse.Namespace:
    """Parse command line"""

    parser = argparse.ArgumentParser()

    parser.add_argument('root_cap', help='tahoe capability URI')
    parser.add_argument('mountpoint', help='mountpoint')
    parser.add_argument('-o', metavar="OPTIONS", help='Mount options', required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # positional arguments
    root_cap = args.root_cap
    mountpoint = args.mountpoint

    # options (defaults)
    node_url = None
    uid = os.getuid()
    gid = os.getgid()
    file_mode = 0o644
    dir_mode = 0o755
    read_only = False
    allow_other = False
    debug = False
    debug_fuse = False
    fork = False
    syslog = False

    for opt in args.o.split(','):
        if '=' in opt:
            (key, value) = opt.split('=')
            if key == 'node_url':
                node_url = value
            elif key == 'setuid':
                uid = value
            elif key == 'setgid':
                gid = value
            elif key == 'file_mode':
                file_mode = int(value, 8)
            elif key == 'dir_mode':
                dir_mode = int(value, 8)
            else:
                print('Ignoring unsupported option:', key)
        else:
            if opt == 'ro':
                read_only = True
            elif opt == 'noexec':
                mode_exec = False
            elif opt == 'exec':
                mode_exec = True
            elif opt == 'allow_other':
                allow_other = True
            elif opt == 'debug':
                debug = True
            elif opt == 'debug_fuse':
                debug_fuse = True
            elif opt == 'fork':
                fork = True
            elif opt == 'nofork':
                fork = False
            elif opt == 'syslog':
                syslog = True
            else:
                print('Ignoring unsupported option:', opt)

    if not node_url:
        print('Specify node_url option')
        exit(1)

    node_urls = node_url.split('+')

    init_logging(syslog, debug)

    log.info('Using settings: node_urls=%s uid=%s gid=%s file_mode=%s dir_mode=%s '
             'read_only=%s allow_other=%s debug=%s debug_fuse=%s fork=%s syslog=%s',
             node_urls, uid, gid, oct(file_mode)[2:], oct(dir_mode)[2:],
             read_only, allow_other, debug, debug_fuse, fork, syslog)

    if fork:
        pid = os.fork()
        if pid == 0:
            pass
        else:
            os._exit(0)  # pylint: disable=protected-access

    testfs = TahoeFs(node_urls, root_cap, read_only, uid, gid, dir_mode, file_mode)
    fuse_options = set(pyfuse3.default_options)
    fuse_options.add('fsname=tahoe')
    if allow_other:
        fuse_options.add('allow_other')
    if debug_fuse:
        fuse_options.add('debug')
    pyfuse3.init(testfs, mountpoint, fuse_options)
    log.info('Initialized successfully')
    try:
        trio.run(pyfuse3.main)
    except BaseException:
        pyfuse3.close(unmount=True)
        raise

    pyfuse3.close()


if __name__ == '__main__':
    main()
