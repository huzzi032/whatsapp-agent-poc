#!/bin/bash
cd /tmp/8de3dadd2366b1c
source antenv/bin/activate
python3 -m gunicorn main:app -c gunicorn.conf.py
