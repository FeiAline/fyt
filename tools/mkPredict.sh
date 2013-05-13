#!/bin/bash
 g++ -o boostingPredict boost_predict.cpp `pkg-config --cflags --libs opencv` -g