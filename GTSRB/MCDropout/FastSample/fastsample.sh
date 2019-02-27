#!/bin/bash 
COUNTER=0
while [  $COUNTER -lt 8 ]; do
	python sample.py
	let COUNTER=COUNTER+1 
done
