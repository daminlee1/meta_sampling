FROM ubuntu:16.04
RUN apt-get update --fix-missing
RUN apt-get install -y python-dev python-pip wget
RUN pip install --upgrade pip
RUN pip install setuptools

# PostgreSQL
RUN /bin/bash -c "echo 'deb http://apt.postgresql.org/pub/repos/apt/ xenial-pgdg main' > /etc/apt/sources.list.d/pgdg.list"
RUN apt-get update
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get update
RUN apt-get install -y postgresql-10

# Django
RUN pip install django

# Adminator
RUN apt-get install -y npm curl
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs

# My project
RUN apt-get install sqlite3 libsqlite3-dev
RUN pip install Pillow
RUN apt-get install python-opencv
