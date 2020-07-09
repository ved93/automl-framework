
#to run this file, use 
#sh run.sh randomforest

export TRAINING_DATA=input/titanic/train_folds.csv
export TEST_DATA=input/titanic/test.csv

export MODEL=$1   #this $1 stands for command line arg i.e.
# sh run.sh randomforest

#check out below for workflow
#https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/workflow.html


# export TRAINING_DATA=input/titanic/train.csv
# export OUTPATH=input/titanic/
# python -m src.cross_validation

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train
python -m src.predict