#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <GITHUB_TOKEN>"
    exit 1
fi

GITHUB_TOKEN=$1

C:\Users\Eric\miniconda3\envs\py39\python.exe -m pip install -U --force-reinstall https://"$GITHUB_TOKEN"@github.com/Nuitka/Nuitka-commercial/archive/main.zip
C:\Users\Eric\miniconda3\envs\py39\python.exe -m nuitka --follow-imports --standalone --enable-plugin=numpy --jobs=22 .\calib.py
