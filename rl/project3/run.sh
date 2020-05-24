#!/bin/bash
#here are all the experiments, in the format of "python3 run.py algo_name init_lr end_lr init_eps end_eps max_iter"

python3 run.py Q_offpolicy 0.15 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_2Q 0.15 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_1Q 0.15 0.001 1.0 0.001 1000000.0 
python3 run.py Q_offpolicy 0.15 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_2Q 0.15 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_1Q 0.15 0.001 1.0 1.0 1000000.0 
python3 run.py foeQ_2Q 0.9 0.001 1.0 0.001 1000000.0 
python3 run.py foeQ_1Q_2LP 0.9 0.001 1.0 0.001 1000000.0 
python3 run.py foeQ_1Q_1LP 0.9 0.001 1.0 0.001 1000000.0 
python3 run.py ceQ 0.9 0.001 1.0 0.001 1000000.0 
python3 run.py foeQ_2Q 0.9 0.001 1.0 1.0 1000000.0 
python3 run.py foeQ_1Q_2LP 0.9 0.001 1.0 1.0 1000000.0 
python3 run.py foeQ_1Q_1LP 0.9 0.001 1.0 1.0 1000000.0 
python3 run.py ceQ 0.9 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_2Q_alt 0.15 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_2Q_alt 0.9 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_2Q_alt 0.5 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_2Q_alt 0.5 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_2Q_alt 0.9 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_2Q_alt 0.15 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_1Q_alt 0.15 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_1Q_alt 0.5 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_1Q_alt 0.9 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_1Q_alt 0.15 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_2Q_alt 0.22 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_2Q_alt 0.22 0.001 1.0 1.0 1000000.0 
python3 run.py friendQ_1Q_alt 0.22 0.001 1.0 0.001 1000000.0 
python3 run.py friendQ_1Q_alt 0.22 0.001 1.0 1.0 1000000.0 




