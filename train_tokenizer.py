import argparse
import itertools
import json
import logging
import pathlib
import sys

import word_list

import torch

from tokenizers import Tokenizer

from tokenizers.models import WordPiece
from tokenizers.models import BPE
from tokenizers.models import Unigram

from tokenizers.trainers import WordPieceTrainer
from tokenizers.trainers import BpeTrainer
from tokenizers.trainers import UnigramTrainer

from torch.nn.utils.rnn import pad_sequence

BOS = "<seq>"
EOS = "</seq>"
PAD = "<pad/>"
UNK = "<unk/>"
EN='<en>'
ES='<es>'
FR='<fr>'
IT='<it>'
RU='<ru>'

def get_parser(
    parser=argparse.ArgumentParser(
        description="run to train a new tokenizer"
    ),
):
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")

    parser.add_argument(
        "--save_corpus",
        # type=pathlib.Path,
        default=pathlib.Path('tokenizers/corpus.txt'),
        help="where to save corpus",
    )
    parser.add_argument(
        "--save_tokenizer",
        # type=pathlib.Path,
        default=pathlib.Path("tokenizers"),
        help="where to save tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=5000000,
        help='maximum size of the new tokenizer'
    )
    parser.add_argument(
        "--reto_type",
        default='WordPiece',
        help='type of tokenizer.Options:BPE, WordPiece, Unigram'
    )
    parser.add_argument(
        "--special_tokens",
        default=[ "<pad/>","</seq>", "<seq>", "<unk/>","<en>","<es>","<fr>","<it>","<ru>"],
        help="format['A','B',...]"
    )
    return parser


def generate_corpus(args):
    train_text = word_list.generate_word_list(args.train_file).gloss_list
    dev_text = word_list.generate_word_list(args.dev_file).gloss_list
    test_text = word_list.generate_word_list(args.test_file).gloss_list
    if dev_text != test_text:
        corpus = train_text+dev_text+test_text
    else:
        corpus = train_text + dev_text
    with open(args.save_corpus, 'w') as target:
        for n in corpus:
            target.write(n+' ')

def train_tokenizer(args):
    from tokenizers.pre_tokenizers import Whitespace
    path = str(args.save_corpus)
    files = [path]
    if args.reto_type=='WordPiece':
            tokenizer = Tokenizer(WordPiece(unk_token="<unk/>"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordPieceTrainer(vocab_size=int(args.vocab_size), special_tokens=args.special_tokens)
            tokenizer.train(files, trainer)
            tokenizer.save(str(args.save_tokenizer)+'/' + f"{args.reto_type}_tokenizer.json")

    elif args.reto_type=='BPE':
            tokenizer = Tokenizer(BPE(unk_token="<unk/>"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer=BpeTrainer(vocab_size=int(args.vocab_size), special_tokens=args.special_tokens)
            tokenizer.train(files, trainer)
            tokenizer.save(str(args.save_tokenizer)+'/' + f"{args.reto_type}_tokenizer.json")

    elif args.reto_type=='Unigram':
            tokenizer = Tokenizer(Unigram())
            tokenizer.pre_tokenizer = Whitespace()
            trainer=UnigramTrainer(vocab_size=int(args.vocab_size), special_tokens=args.special_tokens)
            tokenizer.train(files, trainer)
            tokenizer.save(str(args.save_tokenizer)+'/' + f"{args.reto_type}_tokenizer.json")

if __name__ == "__main__":
    args = get_parser().parse_args()
    generate_corpus(args)
    train_tokenizer(args)