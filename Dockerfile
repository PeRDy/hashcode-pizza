FROM python:3.6-alpine
MAINTAINER PeRDy

ENV APP=hashcode-pizza
ENV WORKDIR=/srv/apps/$APP/app
ENV LOGDIR=/srv/apps/$APP/logs
ENV PYTHONPATH='$PYTHONPATH:$WORKDIR'

# Install system dependencies
RUN apk --no-cache add \
        build-base \
        linux-headers \
        freetype-dev \
        libpng-dev \
        python3-dev && \
    rm -rf /var/cache/apk/*

# Create initial dirs
RUN mkdir -p $WORKDIR $LOGDIR
WORKDIR $WORKDIR

# Install pip requirements
COPY requirements.txt constraints.txt $WORKDIR/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    rm -rf $HOME/.cache/pip/*

# Copy application
COPY . $WORKDIR

ENTRYPOINT ["./__main__.py"]
