import sys
import requests
from urllib.parse import quote
from pathlib import Path
from argparse import ArgumentParser


def main(path: Path, api: str, cap: str, level=0):
    log_prefix = '    ' * level
    print(log_prefix, path, 'uploading children...')
    for sub in path.iterdir():
        name = sub.name
        if sub.is_file():
            r = requests.get(f'{api}/uri/{quote(cap)}/{quote(name)}?t=json')
            if r.status_code == 200:
                print(log_prefix, sub, 'already exists')
                json = r.json()

                if json[0] == 'filenode':
                    if json[1]['format'] == 'CHK':
                        if json[1]['size'] == sub.stat().st_size:
                            print(log_prefix, sub, 'same size')
                            good = True
                        else:
                            print(log_prefix, sub, 'different size')
                            good = False
                    else:
                        print(log_prefix, sub, 'different format', json[1]['format'])
                        good = False
                else:
                    print(log_prefix, sub, 'unexpected node type, probably a directory')
                    good = False

                if good:
                    print(log_prefix, sub, 'skipping')
                else:
                    print(log_prefix, sub, 'deleting...')
                    r = requests.delete(f'{api}/uri/{quote(cap)}/{quote(name)}')
                    assert r.status_code == 200
                    print(log_prefix, sub, 're-uploading...')
                    with open(sub, 'rb') as f:
                        requests.put(f'{api}/uri/{quote(cap)}/{quote(name)}?format=CHK', data=f)
                    print(log_prefix, sub, 'done!')
            elif r.status_code == 404:
                print(log_prefix, sub, 'uploading...')
                with open(sub, 'rb') as f:
                    requests.put(f'{api}/uri/{quote(cap)}/{quote(name)}?format=CHK', data=f)
                print(log_prefix, sub, 'done!')
            else:
                print(log_prefix, 'unexpected status code', r.status_code)
                exit(1)
        elif sub.is_dir():
            r = requests.get(f'{api}/uri/{quote(cap)}/{quote(name)}?t=json')
            if r.status_code == 200:
                json = r.json()
                if json[0] != 'dirnode':
                    print(log_prefix, sub, 'not a directory in tahoe filesystem!')
                else:
                    sub_cap = json[1]['rw_uri']
                    cap_type = sub_cap.split(':')[1]
                    assert cap_type == 'DIR2'
                    print(log_prefix, sub, 'already exists', cap)
                    main(sub, api, sub_cap, level + 1)
            elif r.status_code == 404:
                print(log_prefix, sub, 'creating directory...')
                r = requests.post(f'{api}/uri/{quote(cap)}/{quote(name)}?t=mkdir')
                sub_cap = r.text
                print(log_prefix, sub, 'done!', sub_cap)
                main(sub, api, sub_cap, level + 1)
            else:
                print('unexpected status code', r.status_code)
                exit(1)

    print(log_prefix, path, 'done!')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('path', type=str, help='Path to file or directory to upload')
    parser.add_argument('api', type=str, help='HTTP REST API URL of a Tahoe-LAFS node')
    parser.add_argument('cap', type=str, help='Tahoe directory capability where files should be uploaded to')

    args = parser.parse_args()

    main(Path(args.path), args.api, args.cap)
