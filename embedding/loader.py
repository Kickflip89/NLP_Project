from os import listdir
from os.path import join
from gensim.models import KeyedVectors
from nltk import word_tokenize, sent_tokenize

TEXT_DATA_DIR = './WritingPrompts/'
misspellings = {'*.*.*':'.',
               '*Stupid.*..': 'Stupid',
               'victor.\'..': 'victor',
               'have.*.5': 'have',
               '5.*Sigh..': 'Sigh',
               '*Riiing*ÔÇª.*Riiing*ÔÇª': 'ring',
               '.*IS*..':'is',
               '.*-..':'-',
               'found.\'..':'found',
               '*okay.*..':'okay'}

def get_w2v():
  wv = KeyedVectors.load("./embedding/w2v_128.mdl", mmap='r')
  return wv

class StoryDataset(torch.utils.data.IterableDataset):
    def __init__(self, fold, d_path, w2v, prompt_len, seq_len):
        pname = f'{fold}.wp_source'
        sname = f'{fold}.wp_target'
        self.p_path = join(d_path, pname)
        self.s_path = join(d_path, sname)
        self.fold = fold
        self.w2v = w2v
        self.prompt_len = prompt_len
        self.seq_len = seq_len
        
    def get_embedding_array(self, line):
        all_words = []
        words = word_tokenize(line)
        for word in words:
            if word in misspellings:
                word = misspellings[word]
            try:
                all_words.append(self.w2v[word][None,:])
            except Exception as e:
                all_words.append(self.w2v[' '])[None,:]
                print(f'error with {word}, {e}')
        return np.concatenate(all_words, axis=0)
        
    def make_prompt(self, line):
        p = self.get_embedding_array(line)
        pad = np.zeros((self.prompt_len - p.shape[0], self.w2v.wv.vector_size))
        return np.concatenate([pad,p])
        
    def make_story(self, line):
        return self.get_embedding_array(line)
        
    def make_data(self, p_line, s_line):
        p = self.make_prompt(p_line)
        s = self.make_story(s_line)
        i = np.random.randint(0,s.shape[0]-self.seq_len-1)
        seq = s[i:i+self.seq_len].copy()
        y = s[i+self.seq_len].copy()
        res = self.w2v.similar_by_vector(y)[0][0]
        y = np.array([self.w2v.wv.vocab.get(res).index])
        p = torch.tensor(p).float()
        s = torch.tensor(seq).float()
        y = torch.tensor(y).long().squeeze(-1)
        return p,s,y
    
    def __iter__(self):
        self.seq_len = np.random.randint(20,100)
        print(self.seq_len)
        p_iter = open(self.p_path, 'r', encoding='cp850')
        s_iter = open(self.s_path, 'r', encoding='cp850')
        
        tensor_iter = map(self.make_data, p_iter, s_iter)
        
        return tensor_iter
