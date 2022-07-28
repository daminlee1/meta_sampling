FROM ubuntu:16.04
RUN apt-get update --fix-missing
RUN apt-get install -y python-dev python-pip wget npm curl
RUN pip install --upgrade pip
RUN pip install setuptools

# Django
RUN pip install django

# My project
RUN apt-get install -y sqlite3 libsqlite3-dev
RUN pip install Pillow requests
RUN apt-get install -y python-opencv

# pycocotools
RUN pip install --upgrade pip
RUN apt-get install -y libfreetype6-dev pkg-config libpng12-dev git
