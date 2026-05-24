#!/bin/bash
# lab1/pipeline.sh

# Директива прерывания при возникновении ошибки
set -e 

echo ">>> Starting MLOps Pipeline (Module 1) <<<"


#python3 -m venv venv
#source venv/bin/activate
pip install pandas scikit-learn numpy

echo "Step 1: Data Creation"
python3 lab1/data_creation.py

echo "Step 2: Data Preprocessing"
python3 lab1/data_preprocessing.py

echo "Step 3: Model Preparation & Training"
python3 lab1/model_preparation.py

echo "Step 4: Model Testing & Evaluation"
# Вывод последней строки будет содержать оценку метрики
python3 lab1/model_testing.py

echo ">>> Pipeline executed successfully. <<<"