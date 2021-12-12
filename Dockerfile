FROM python:3.9-bullseye AS builder

RUN apt-get update

RUN pip install pyinstaller

RUN apt-get install -y libfuse3-dev

COPY requirements.txt /requirements
RUN pip install -r /requirements

WORKDIR /data
COPY . .

RUN pyinstaller --name tahoe-mount mount.py

ENTRYPOINT ["bash", "/entrypoint.sh"]

FROM debian:bullseye-slim

COPY --from=builder /data/dist/tahoe-mount /tahoe-mount

ENTRYPOINT ["/tahoe-mount/tahoe-mount"]
