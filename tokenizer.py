import os
import json
import pickle
from absl import app, flags


class Tokenizer:
    def __init__(self, questions=None):
        """
        :param questions: List[str]
        """
        if questions:
            print(len(questions), 'questions')
            tokens = set()
            for x in questions:
                tokens.update(x.lower().split(' '))
            del questions
            tokens = list(tokens)
            tokens.sort()
            self.id2word = dict(enumerate(tokens, start=2))
            self.word2id = {v: k for k, v in self.id2word.items()}
            self.unk_id = 0
            self.id2word[self.unk_id] = 'UNK'
            self.word2id['UNK'] = self.unk_id
            self.pad_id = 1
            self.id2word[self.pad_id] = 'PAD'
            self.word2id['PAD'] = self.pad_id
        else:
            self.id2word = None
            self.word2id = None
            self.unk_id = None

    def encode_sentence(self, sent):
        return [self.word2id.get(x, self.unk_id) for x in sent.lower().split(' ')]

    def decode_sentence(self, ids):
        return [self.id2word.get(x, '??') for x in ids]

    def add_special(self, word):
        new_id = max(self.id2word.keys()) + 1
        self.id2word[new_id] = word
        self.word2id[word] = new_id

    def get_vocab_size(self):
        return len(self.word2id) if self.word2id else 0

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            state = {
                'id2word': self.id2word,
                'word2id': self.word2id
            }
            pickle.dump(state, f)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            state = pickle.load(f)
        self.word2id = state['word2id']
        self.id2word = state['id2word']
        self.unk_id = self.word2id['UNK']
        self.pad_id = self.word2id['PAD']
        return self


class LabelEncoder:
    TRAIN_LABELS = {'0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'blue', 'brown', 'cube',
                    'cyan', 'cylinder', 'gray', 'green', 'large', 'metal', 'no', 'purple', 'red',
                    'rubber', 'small', 'sphere', 'yellow', 'yes'}

    def __init__(self, labels):
        labels = sorted(list(labels))
        self.id2label = dict(enumerate(labels))
        self.label2id = {v: k for k, v in self.id2label.items()}

    def encode(self, label):
        return self.label2id[label]

    def decode(self, id_):
        return self.id2label[id_]


def main(argv):
    FLAGS = flags.FLAGS
    questions = json.load(open(FLAGS.json_path, 'r'))['questions']
    t = Tokenizer([x['question'].strip('?') for x in questions])
    # print(t.encode_sentence("hello many big objects"))
    t.save(FLAGS.save_to)
    # t = Tokenizer().load('tmp/tiny-tokenizer.pickle')
    # print(t.word2id)
    print('vocab size', t.get_vocab_size())

    en = LabelEncoder(set([x['answer'] for x in questions]))
    print(en.id2label.values())


if __name__ == '__main__':
    flags.DEFINE_string('json_path', 'data/CLEVR_v1.0/questions/CLEVR_tiny_val_questions.json',
                        'path to read question json')
    flags.DEFINE_string('save_to', 'tmp/tiny-tokenizer.pickle', 'path to save tokenizer')
    app.run(main)
