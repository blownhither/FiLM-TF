import json
import pickle


class Tokenizer:
    def __init__(self, question_file=None):
        if question_file:
            questions = json.load(open(question_file, 'r'))['questions']
            tokens = set()
            for x in questions:
                tokens.update(x['question'].strip('?').lower().split(' '))
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
    TINY_VAL_LABELS = {'cyan', 'rubber', 'brown', 'sphere', 'yes', '0', '1', 'no', 'metal',
                       'purple', '2', '6', 'yellow'}

    def __init__(self, labels):
        labels = sorted(list(labels))
        self.id2label = dict(enumerate(labels))
        self.label2id = {v: k for k, v in self.id2label.items()}

    def encode(self, label):
        return self.label2id[label]

    def decode(self, id_):
        return self.id2label[id_]


if __name__ == '__main__':
    t = Tokenizer('data/CLEVR_v1.0/questions/CLEVR_tiny_val_questions.json')
    print(t.encode_sentence("hello many big objects"))
    t.save('tmp/tiny-tokenizer.pickle')
    t = Tokenizer().load('tmp/tiny-tokenizer.pickle')
    print(t.word2id)
    print('vocab size', t.get_vocab_size())


