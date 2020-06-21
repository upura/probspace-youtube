#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/drive/16FxkNDXis5dDkJA0RHZu9x9lyxkY-y3s?authuser=1
  sleep 3600
done
