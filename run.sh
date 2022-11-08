#!/bin/sh

if [ $1 == "s" ];
then
    python3 Perceptron/standard_perceptron.py $2 $3
fi
if [ $1 == "v" ];       
then
    python3 Perceptron/voted_perceptron.py $2 $3
fi
if [ $1 == "a" ];
then
    python3 Perceptron/average_perceptron.py $2 $3
fi
