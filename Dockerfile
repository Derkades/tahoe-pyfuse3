FROM python:3.9-bullseye AS builder

RUN apt-get update

RUN pip install pyinstaller

RUN apt-get install -y libfuse3-dev
RUN pip install pyfuse3 urllib3

WORKDIR /data
COPY . .

RUN pyinstaller --onefile --name tahoe-mount mount.py

ENTRYPOINT ["bash", "/entrypoint.sh"]

FROM debian:bullseye-slim

COPY --from=builder /data/dist/tahoe-mount /tahoe-mount

ENTRYPOINT ["/tahoe-mount"]
