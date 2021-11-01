import sys
import requests
from urllib.parse import quote
from pathlib import Path
from argparse import ArgumentParser


def upload_file(path: Path, api: str, parent_cap: str, log_prefix: str):
    with open(path, 'rb') as f:
        r = requests.put(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?format=CHK', data=f)
        if r.status_code == 201:
            print(log_prefix, sub, 'done!')
        else:
            print(r.text)
            raise Exception('Failed to upload file')


def check_upload_file(path: Path, api: str, parent_cap: str, log_prefix: str):
    # check if a file or directory with this name already exists
    r = requests.get(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?t=json')
    if r.status_code == 200:
        print(log_prefix, path, 'already exists')
        json = r.json()

        if json[0] == 'filenode':
            if json[1]['format'] == 'CHK':
                if json[1]['size'] == path.stat().st_size:
                    print(log_prefix, path, 'same size, skipping')
                    return
                else:
                    print(log_prefix, path, 'different size')
            else:
                print(log_prefix, path, 'different format', json[1]['format'])
        else:
            print(log_prefix, path, 'unexpected node type (probably a directory)')

        print(log_prefix, path, 'deleting...')
        r = requests.delete(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}')
        assert r.status_code == 200
        print(log_prefix, path, 're-uploading...')
        upload_file(path, )
    elif r.status_code == 404:
        print(log_prefix, path, 'uploading...')
        upload_file(path, api, parent_cap, log_prefix)
    else:
        raise Exception('Unexpected status code ' + r.status_code)


def upload_dir(path: Path, api: str, parent_cap: str, log_prefix: str, level: int):
    r = requests.get(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?t=json')
    if r.status_code == 200:
        json = r.json()
        if json[0] != 'dirnode':
            print(log_prefix, path, 'not a directory in tahoe filesystem!')
        else:
            cap = json[1]['rw_uri']
            cap_type = cap.split(':')[1]
            assert cap_type == 'DIR2'
            print(log_prefix, path, 'already exists', cap)
    elif r.status_code == 404:
        print(log_prefix, path, 'creating directory...')
        r = requests.post(f'{api}/uri/{quote(parent_cap)}/{quote(path.name)}?t=mkdir')
        cap = r.text
        print(log_prefix, path, 'created', cap)
    else:
        raise Exception('Unexpected status code ' + r.status_code)

    print(log_prefix, parent_path, 'uploading contents....')
    upload_contents(parent_path=path, api=api, parent_cap=cap, level=level+1)
    print(log_prefix, parent_path, 'done!')


def upload_contents(parent_path: Path, api: str, level: int):
    log_prefix = '    ' * level
    for path in parent_path.iterdir():
        if path.is_file():
            check_upload_file(path, api, parent_cap, log_prefix)
        elif path.is_dir():
            upload_dir(path, api, parent_cap, log_prefix, level)
        else:
            print(log_prefix, path, "skipping, unknown file type")
    

def main(path: Path, api: str, cap: str, contents_only: bool):
    if contents_only:
        upload_contents(parent_path=path, api=api, parent_cap=cap, level=0)
    else:
        upload_dir(path=path, api=api, parent_cap=cap, log_prefix='', level=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('path', type=str, help='Path to file or directory to upload')
    parser.add_argument('api', type=str, help='HTTP REST API URL of a Tahoe-LAFS node')
    parser.add_argument('cap', type=str, help='Tahoe directory capability where files should be uploaded to')
    parser.add_argument('contents-only', type=bool, 
                        help='When set to true, only the directory\'s contents will be '
                             'stored. When set to false, the directory itself will be uploaded.')

    args = parser.parse_args()

    main(path=Path(args.path), api=args.api, cap=args.cap, contents_only=args.contents_only)
