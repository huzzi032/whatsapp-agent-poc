#!/bin/bash
set -e

cd /home/site/wwwroot

# activate Azure-created venv
source antenv/bin/activate

# start app
python -m gunicorn main:app -c gunicorn.conf.py
