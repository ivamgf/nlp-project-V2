#!/bin/bash

# Caminho completo para os arquivos Python
python_script_1="/LSTM-Model-01.py"

python_script_2="/BILSTM-CE-Model-01.py"

python_script_3="/BILSTM-CE-Model-CV-01.py"

# Executa os arquivos Python
python3 "$python_script_1"

python3 "$python_script_2"

python3 "$python_script_3"
