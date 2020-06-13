export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv

export MODEL=$1

#check out below for workflow
#https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/workflow.html

python -m src.cross_validation

# FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train
python -m src.predict