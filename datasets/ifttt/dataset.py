# coding=utf-8
from __future__ import print_function
from absl import app
from absl import flags
import sys
import numpy as np
try: import cPickle as pickle
except: import pickle
sys.path.append("/home/xbb/tranX")
from asdl.hypothesis import Hypothesis
from components.action_info import get_action_infos
from asdl.transition_system import *
from components.dataset import Example
from components.vocab import VocabEntry, Vocab
from asdl.lang.ifttt.ifttt_dcs_transition_system import *
from asdl.lang.ifttt.ifttt_form import *
from copy import deepcopy
level_traversal=False
right_left = False
def load_dataset(transition_system, dataset_file, reorder_predicates=True,return_golden=False):
    examples = []
    for idx, line in enumerate(open(dataset_file)):
        src_query, tgt_code,tag = line.strip().split('||&||')
        tgt_code = tgt_code.replace("  "," ")
        src_query_tokens = src_query.split(' ')
        if return_golden:
            if tag != "intelligible gold english":
                continue

        lf = ifttt_ast_to_parse_tree(tgt_code)
        assert lf.to_string() == tgt_code

        gold_source = lf.to_string()
        grammar = transition_system.grammar
        tgt_ast = ifttt_form_to_ast(grammar, lf)
        reconstructed_lf = ast_to_ifttt_form(tgt_ast)
        assert lf == reconstructed_lf

        tgt_actions = transition_system.get_actions(tgt_ast,level_traversal=level_traversal,right_left=right_left)
        tgt_action_infos = get_action_infos(src_query_tokens, tgt_actions,level_traversal=level_traversal,right_left=right_left)
        
        example = Example(idx=idx,
              src_sent=src_query_tokens,
              tgt_actions=tgt_action_infos,
              tgt_code=gold_source,
              tgt_ast=tgt_ast,
              meta=None)
        '''
        print("id:",idx)
        print("src_sent:",src_query_tokens)
        print("tgt_actions:",tgt_action_infos)
        print("tgt_code:",gold_source)
        print("tgt_ast:",tgt_ast)
        print("**********************************************")
        '''
        examples.append(example)

    return examples


def prepare_ifttt_dataset():
    vocab_freq_cutoff = 2
    grammar = ASDLGrammar.from_text(open('asdl/lang/ifttt/ifttt_asdl.txt').read())
    transition_system = IFTTTTransitionSystem(grammar)

    train_set = load_dataset(transition_system, 'data/ifttt/train.txt')
    dev_set = load_dataset(transition_system, 'data/ifttt/dev.txt')
    test_set = load_dataset(transition_system, 'data/ifttt/test.txt',return_golden=True)

    test_action_len = []
    
    split_num = 10
    max_num = 40
    times = int(max_num / split_num)
    data_split = {}
    for i in range(times+1):
        data_split[str(i*10+1)+"-"+str(i*10+10)] = []
    
    for example in test_set:
        if len(example.tgt_actions) <= 110:
            test_action_len.append(len(example.tgt_actions))
           
            index = min(int(len(example.tgt_actions) / 10),4)
            data_split[str(index*10+1)+"-"+str(index*10+10)].append(example)
           
        '''
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from collections import Counter
    plt.switch_backend('Agg')
    pdf = PdfPages('cut_figure.pdf')         #先创建一个pdf文件
    plt.figure()
    count = Counter(test_action_len)
    x = sorted(count)
    y = []
    for i in sorted(count):
        y.append(count[i])

    plt.plot(x, y)

    plt.title('ifttt')
    plt.xlabel('length')
    plt.ylabel('count')
    plt.savefig("data/ifttt/atis.png")
    plt.close()
    pdf.close()  
    '''
    for i in data_split:
        print(i,": num=",len(data_split[i]))
        pickle.dump(data_split[i], open('data/ifttt/data_split/'+i+".bin", 'wb'))



    
    
    
    
    # generate vocabulary
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_set], size=30000, freq_cutoff=vocab_freq_cutoff)

    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_set]

    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=50000, freq_cutoff=0)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_set]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=4000, freq_cutoff=2)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, dev_set, test_set)]
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    
    
    file_name = ""
    if level_traversal:
        file_name = 'data/ifttt/train_teacher.bin'
    elif right_left:
        file_name = 'data/ifttt/train_r2l.bin'
    else:
        file_name = 'data/ifttt/train_student.bin' 


    pickle.dump(train_set, open(file_name , 'wb'))
    pickle.dump(dev_set, open('data/ifttt/dev.bin', 'wb'))
    pickle.dump(test_set, open('data/ifttt/test.bin', 'wb'))
    pickle.dump(vocab, open('data/ifttt/vocab.freq%d.bin' % vocab_freq_cutoff, 'wb'))
    print("train:", len(train_set))
    print("test:", len(test_set))
    print("dev:", len(dev_set))


def generate_vocab_for_paraphrase_model(vocab_path, save_path):
    from components.vocab import VocabEntry, Vocab

    vocab = pickle.load(open(vocab_path))
    para_vocab = VocabEntry()
    for i in range(0, 10):
        para_vocab.add('<unk_%d>' % i)
    for word in vocab.source.word2id:
        para_vocab.add(word)
    for word in vocab.code.word2id:
        para_vocab.add(word)

    pickle.dump(para_vocab, open(save_path, 'w'))


def main(arg):
    grammar = ASDLGrammar.from_text(open('asdl/lang/ifttt/ifttt_asdl.txt').read())
    transition_system = IFTTTTransitionSystem(grammar)
    prepare_ifttt_dataset()

if __name__ == '__main__':
    app.run(main)
