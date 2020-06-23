#!/bin/bash

for i in `seq 0 24`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/drive/1EUU9jQj8ZZTXo1UyesyJGs-0-p8Zc_nC?authuser=1
  sleep 3600
done
