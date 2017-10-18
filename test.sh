#!/bin/bash

python tf_sm.py --mode test
gzip tf_submit_sm.csv
sudo cp tf_submit_sm.csv.gz /var/www/html

