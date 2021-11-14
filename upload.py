import sys
import requests
from urllib.parse import quote
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


# TODO make configurable through command flags
UPLOAD_BLOCK_SIZE = 512*1024
UPLOAD_PROGRESS_UNIT = 'bytes'
assert UPLOAD_BLOCK_SIZE > 0
assert UPLOAD_PROGRESS_UNIT in {'bytes', 'bits'}


def file_reader(f, bar):
    while True:
        data = f.read(UPLOAD_BLOCK_SIZE)
        if data:
            yield data
            bar.update(len(data) * (8 if UPLOAD_PROGRESS_UNIT == 'bits' else 1))
        else:
            break


def upload_file(path: Path, file_size: int, api: str, parent_cap: str, log_prefix: str):
    with open(path, 'rb') \
            as f, tqdm(desc=(log_prefix + path.name),
                       total=file_size * (8 if UPLOAD_PROGRESS_UNIT == 'bits' else 1),
                       unit='iB' if UPLOAD_PROGRESS_UNIT == 'bytes' else 'ib',
                       unit_scale=True,
                       unit_divisor=1024) as bar:
        r = requests.put(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?format=CHK',
                         data=file_reader(f, bar),
                         headers={'Accept': 'text/plain'})

    if r.status_code == 201:
        print(log_prefix + path.name, 'done!')
    else:
        print(r.text)
        print('Failed to upload file ' + str(path))
        exit(1)


def check_upload_file(path: Path, api: str, parent_cap: str, log_prefix: str):
    # check if a file or directory with this name already exists
    print(log_prefix + path.name, end=': ')
    r = requests.get(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?t=json',
                     headers={'Accept': 'text/plain'})
    if r.status_code == 200:
        print('already exists, ', end='', flush=True)
        json = r.json()

        if json[0] == 'filenode':
            if json[1]['format'] == 'CHK':
                file_size = path.stat().st_size
                if json[1]['size'] == file_size:
                    print('same size',)
                    return
                else:
                    print('different size', end=' ', flush=True)
            else:
                print('different format', json[1]['format'], end=' ', flush=True)
        else:
            print('unexpected node type (probably a directory)', end=' ', flush=True)

        print('deleting...', end=' ', flush=True)
        r = requests.delete(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}',
                            headers={'Accept': 'text/plain'})
        if r.status_code != 200:
            print('unexpected status code', r.status_code)
            print(r.text)
            exit(1)
        print('re-uploading...')
        upload_file(path, file_size, api, parent_cap, log_prefix)
    elif r.status_code == 404:
        print('uploading...')
        upload_file(path, path.stat().st_size, api, parent_cap, log_prefix)
    else:
        print(r.text)
        print('Unexpected status code ' + str(r.status_code))
        exit(1)


def upload_dir(path: Path, api: str, parent_cap: str, log_prefix: str):
    print(log_prefix + path.name, end=': ')
    r = requests.get(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?t=json',
                     headers={'Accept': 'text/plain'})
    if r.status_code == 200:
        json = r.json()
        if json[0] != 'dirnode':
            print('not a directory in tahoe filesystem!')
            exit(1)
        else:
            cap = json[1]['rw_uri']
            cap_type = cap.split(':')[1]
            assert cap_type == 'DIR2'
            print('already exists, ', end='', flush=True)
    elif r.status_code == 404:
        print('creating directory...', end=' ', flush=True)
        r = requests.post(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?t=mkdir',
                          headers={'Accept': 'text/plain'})
        cap = r.text
        print('created,', end=' ', flush=True)
    else:
        print(r.text)
        print('Unexpected status code ' + str(r.status_code))
        exit(1)

    print('uploading contents....')
    upload_contents(parent_path=path, api=api, parent_cap=cap, log_prefix=(log_prefix + '    '))


def upload_contents(parent_path: Path, api: str, parent_cap: str, log_prefix: str):
    for path in parent_path.iterdir():
        if path.is_file():
            check_upload_file(path, api, parent_cap, log_prefix)
        elif path.is_dir():
            upload_dir(path, api, parent_cap, log_prefix)
        else:
            print(log_prefix + path.name, "skipping, unknown file type")


def main(path_str: str, api: str, cap: str):
    path = Path(path_str)
    if path_str.endswith('/'):
        upload_contents(parent_path=path, api=api, parent_cap=cap, log_prefix='')
    else:
        upload_dir(path=path, api=api, parent_cap=cap, log_prefix='')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('path', type=str, help='Path to file or directory to upload. Like rsync, add a trailing slash '
                                               'to upload directory contents, no trailing slash to upload the '
                                               'directory itself.')
    parser.add_argument('api', type=str, help='HTTP REST API URL of a Tahoe-LAFS node')
    parser.add_argument('cap', type=str, help='Tahoe directory capability where files should be uploaded to')
    args = parser.parse_args()

    main(path_str=args.path, api=args.api, cap=args.cap)
