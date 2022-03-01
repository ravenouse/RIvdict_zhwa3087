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

class corpus:
    def __init__(self,train_path,dev_path,test_path,save='corpus_for_reto.txt'):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.save = save
        # self.generate_corpus()

    def generate_corpus(self):
        train_text = word_list.generate_word_list(self.train_path).gloss_list
        dev_text = word_list.generate_word_list(self.dev_path).gloss_list
        test_text = word_list.generate_word_list(self.test_path).gloss_list
        if dev_text != test_text:
            corpus = train_text+dev_text+test_text
        else:
            corpus = train_text + dev_text
        with open(self.save, 'w') as target:
            for n in corpus:
                target.write(n+' ')


class reto:
    def __init__(self,corpus_path='corpus_for_reto.txt',vocab_size=80000,
                 special_tokens= [ "<pad/>","</seq>", "<seq>", "<unk/>"],type='WordPiece',
                 save='tokenizer.json'
                 ):
        # self.generate_corpus()
        self.corpus_path =corpus_path
        self.special_tokens=special_tokens
        self.vocab_size = vocab_size
        self.reto_type=type
        self.save=save


    def train_tokenizer(self):
        from tokenizers.pre_tokenizers import Whitespace
        files = [self.corpus_path]
        if self.reto_type=='WordPiece':
            tokenizer = Tokenizer(WordPiece(unk_token="<unk/>"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
            tokenizer.train(files, trainer)
            tokenizer.save(self.save)

        elif self.reto_type=='BPE':
            tokenizer = Tokenizer(BPE(unk_token="<unk/>"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer=BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
            tokenizer.train(files, trainer)
            tokenizer.save(self.save)

        elif self.reto_type=='Unigram':
            tokenizer = Tokenizer(Unigram())
            tokenizer.pre_tokenizer = Whitespace()
            trainer=UnigramTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
            tokenizer.train(files, trainer)
            tokenizer.save(self.save)

    def encode_batch(self,batch_object):
        tokenizer = Tokenizer.from_file(self.save)
        new_vocab = tokenizer.get_vocab()
        new_gloss_tensor = []
        output = tokenizer.encode_batch(batch_object["gloss"])
        for n in range(len(output)):
            one_gloss = [new_vocab[BOS]] + output[n].ids + [new_vocab[EOS]]
            one_gloss_tensor = torch.tensor(one_gloss)
            new_gloss_tensor.append(one_gloss_tensor)
        new_gloss_tensor = pad_sequence(new_gloss_tensor, batch_first=True, padding_value=new_vocab[PAD]).T
        batch_object["new_gloss_tensor"] = new_gloss_tensor
        return batch_object

    def get_vocab(self):
        tokenizer = Tokenizer.from_file(self.save)
        new_vocab=tokenizer.get_vocab()
        return new_vocab



# train = 'train-data_all/en.train.json'
# dev = "train-data_all/en.dev.json"
# test = dev
#
# # corpus(train_path=train,dev_path=dev,test_path=test)
# tokenizer= reto(type='BPE')
# tokenizer.train_tokenizer()
# new_vocab = tokenizer.get_vocab()


