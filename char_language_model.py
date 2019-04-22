from collections import Counter
SOS_token = 0
EOS_token = 1


class Lang:
    '''
    a language class for tokenized dataset and char model'''
    def __init__(self, name):
        self.name = name
        self.char2index = None
        self.char2count = None
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS
        self.corpus = None

    def build_corpus(self, lines):
        '''build corpus, prepare maps'''
        corpus = ''
        new_lines = []
        for line in lines:
            # each element in one line is bytes, need to be decoded
            concatenated_line = self.decode_line2string(line)
            new_lines.append(concatenated_line)
            corpus+=concatenated_line
        self.corpus = corpus
        self.prepare_charmap()
        return new_lines
    
    def prepare_charmap(self):
        '''get char2index, index2char map from corput'''
        self.char2count = Counter(self.corpus)
        chars = list(self.char2count.keys())
        chars.sort()
        self.n_chars += len(chars)
        self.char2index= {self.index2char[key]: key for key in self.index2char}
        self.char2index.update({c: i+2 for i, c in enumerate(chars)})
        self.index2char = {self.char2index[key]: key for key in self.char2index}
    
    @ staticmethod
    def decode_line2string(line):
        '''decode bytes list of words to a string'''
        return ' '.join(map(lambda x: x.decode('utf-8'), line))
    
    def string2indexes(self, line_string):
        '''encode a line string to indexes, add EOS'''
        indexes = None
        try:
            indexes = [self.char2index[char] for char in line_string]
            indexes.append(EOS_token)
        except KeyError:
            print('There exists a char that is not in the corpus mapping')
        return indexes