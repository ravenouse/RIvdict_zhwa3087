#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=mpi-job
#SBATCH --output=mpi-job.%j.out

source /curc/sw/anaconda3/latest
conda activate mycustomenv

##en
python revdict.py --do_pred \
--train_file ../../data/train_and_dev/en.train.json \
--dev_file ../../data/train_and_dev/en.dev.json \
--test_file ../../data/train_and_dev/en.dev.json \
--tokenizer ../../tokenizers/en/WordPiece_tokenizer.json \
--target_arch 'sgns' \
--summary_logdir ../logs/en/sgns \
--save_dir ../models/en/sgns \
--pred_dir ../predictions/en/predictions1.json

python revdict.py --do_pred \
--train_file ../../data/train_and_dev/en.train.json \
--dev_file ../../data/train_and_dev/en.dev.json \
--test_file ../../data/train_and_dev/en.dev.json \
--tokenizer ../../tokenizers/en/WordPiece_tokenizer.json \
--target_arch 'char' \
--summary_logdir ../logs/en/char \
--save_dir ../models/en/char \
--pred_dir ../predictions/en/predictions2.json

python revdict.py --do_pred \
--train_file ../../data/train_and_dev/en.train.json \
--dev_file ../../data/train_and_dev/en.dev.json \
--test_file ../../data/train_and_dev/en.dev.json \
--tokenizer ../../tokenizers/en/WordPiece_tokenizer.json \
--target_arch 'electra' \
--summary_logdir ../logs/en/electra \
--save_dir ../models/en/electra \
--pred_dir ../predictions/en/predictions3.json

##es
python revdict.py --do_pred \
--train_file ../../data/train_and_dev/es.train.json \
--dev_file ../../data/train_and_dev/es.dev.json \
--test_file ../../data/train_and_dev/es.dev.json \
--tokenizer ../../tokenizers/es/WordPiece_tokenizer.json \
--target_arch 'sgns' \
--summary_logdir ../logs/es/sgns \
--save_dir ../models/es/sgns \
--pred_dir ../predictions/es/predictions1.json

python revdict.py --do_pred \
--train_file ../../data/train_and_dev/es.train.json \
--dev_file ../../data/train_and_dev/es.dev.json \
--test_file ../../data/train_and_dev/es.dev.json \
--tokenizer ../../tokenizers/es/WordPiece_tokenizer.json \
--target_arch 'char' \
--summary_logdir ../logs/es/char \
--save_dir ../models/es/char \
--pred_dir ../predictions/es/predictions2.json


##it
python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/it.train.json \
--dev_file ../../data/train_and_dev/it.dev.json \
--test_file ../../data/train_and_dev/it.dev.json \
--tokenizer ../../tokenizers/it/WordPiece_tokenizer.json \
--target_arch 'sgns' \
--summary_logdir ../logs/it/sgns \
--save_dir ../models/it/sgns \
--pred_dir ../predictions/it/predictions1.json

python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/it.train.json \
--dev_file ../../data/train_and_dev/it.dev.json \
--test_file ../../data/train_and_dev/it.dev.json \
--tokenizer ../../tokenizers/it/WordPiece_tokenizer.json \
--target_arch 'char' \
--summary_logdir ../logs/it/char \
--save_dir ../models/it/char \
--pred_dir ../predictions/it/predictions2.json


##fr
python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/fr.train.json \
--dev_file ../../data/train_and_dev/fr.dev.json \
--test_file ../../data/train_and_dev/fr.dev.json \
--tokenizer ../../tokenizers/fr/WordPiece_tokenizer.json \
--target_arch 'sgns' \
--summary_logdir ../logs/fr/sgns \
--save_dir ../models/fr/sgns \
--pred_dir ../predictions/fr/predictions1.json

python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/fr.train.json \
--dev_file ../../data/train_and_dev/fr.dev.json \
--test_file ../../data/train_and_dev/fr.dev.json \
--tokenizer ../../tokenizers/fr/WordPiece_tokenizer.json \
--target_arch 'char' \
--summary_logdir ../logs/fr/char \
--save_dir ../models/fr/char \
--pred_dir ../predictions/fr/predictions2.json

python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/fr.train.json \
--dev_file ../../data/train_and_dev/fr.dev.json \
--test_file ../../data/train_and_dev/fr.dev.json \
--tokenizer ../../tokenizers/fr/WordPiece_tokenizer.json \
--target_arch 'electra' \
--summary_logdir ../logs/fr/electra \
--save_dir ../models/fr/electra \
--pred_dir ../predictions/fr/predictions3.json


##ru
python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/ru.train.json \
--dev_file ../../data/train_and_dev/ru.dev.json \
--test_file ../../data/train_and_dev/ru.dev.json \
--tokenizer ../../tokenizers/ru/WordPiece_tokenizer.json \
--target_arch 'sgns' \
--summary_logdir ../logs/ru/sgns \
--save_dir ../models/ru/sgns \
--pred_dir ../predictions/ru/predictions1.json

python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/ru.train.json \
--dev_file ../../data/train_and_dev/ru.dev.json \
--test_file ../../data/train_and_dev/ru.dev.json \
--tokenizer ../../tokenizers/ru/WordPiece_tokenizer.json \
--target_arch 'char' \
--summary_logdir ../logs/ru/char \
--save_dir ../models/ru/char \
--pred_dir ../predictions/ru/predictions2.json

python revdict.py --do_train --do_pred \
--train_file ../../data/train_and_dev/ru.train.json \
--dev_file ../../data/train_and_dev/ru.dev.json \
--test_file ../../data/train_and_dev/ru.dev.json \
--tokenizer ../../tokenizers/ru/WordPiece_tokenizer.json \
--target_arch 'electra' \
--summary_logdir ../logs/ru/electra \
--save_dir ../models/ru/electra \
--pred_dir ../predictions/ru/predictions3.json