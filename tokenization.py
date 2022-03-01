# import torchnlp
import data
import word_list
# from torchnlp.encoders.text import SubwordEncoder
#
#
# data = word_list.generate_word_list('/Users/eric/PycharmProjects/CSCI5832/FINAL/train-data_all/en.dev.json')
# texts = data.gloss_list
#
# from tokenizers import Tokenizer,models,trainers
# tokenizer = Tokenizer(models.BPE())
#
# trainer = trainers.BpeTrainer(
#     vocab_size=20000,
#     special_tokens=["<pad>", "<unk>", "</seq>","<seq>"],
# )
#
#
# tokenizer.train_from_iterator(texts, trainer=trainer)
# result =[]
# for n in range(10):
#     tokens = tokenizer.encode(texts[n])
#     result.append(tokens)



# tokenizer = SubwordEncoder(sample=texts,append_eos=True,eos_index=1)


# train_dataset = data.JSONDataset('/Users/eric/PycharmProjects/CSCI5832/FINAL/train-data_all/en.dev.json')
#

# tokenizer = SubwordEncoder(sample=texts,target_vocab_size=20000)


data = word_list.generate_word_list('/Users/eric/PycharmProjects/CSCI5832/FINAL/train-data_all/en.dev.json')
texts = data.gloss_list
# with open ('corpus.txt','w') as object:
#     for n in texts:
#         object.write(n+'\n')
import sentencepiece as spm
# spm.SentencePieceTrainer.Train('--input=corpus.txt --model_prefix=m --vocab_size=6602')
sp = spm.SentencePieceProcessor()
sp.Load("m.model")
vocab ={}
for n in range(sp.get_piece_size()):
    vocab[sp.id_to_piece(n)] = n