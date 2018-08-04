#!/bin/sh
git log --diff-filter=D --summary
git checkout $commit~1 skempi.ipynb
