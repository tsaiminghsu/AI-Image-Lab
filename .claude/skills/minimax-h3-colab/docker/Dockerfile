# google-colab-cli supports Linux and macOS only (its console module imports termios at startup), so on
# Windows the runner invokes it through this image. ENTRYPOINT is the CLI itself:
#   docker run --rm -v h3-colab-config:/root/.config/colab-cli h3-colab-cli:0.7.4 --auth=oauth2 usage
# The OAuth token and session metadata live in that named volume, never in the image or the repo.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "google-colab-cli==0.7.4"

WORKDIR /work
ENTRYPOINT ["colab"]
