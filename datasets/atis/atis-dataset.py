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
from asdl.lang.lambda_dcs.lambda_dcs_transition_system import *
from asdl.lang.lambda_dcs.logical_form import *
from copy import deepcopy
FLAGS = flags.FLAGS
level_traversal=False
right_left = True
flags.DEFINE_string("vocab_freq_cutoff","30","vocab_freq_cutoff")
flags.DEFINE_string("pre_model","transformer","the specific pre for model")

def load_dataset(transition_system, dataset_file, reorder_predicates=True):
    examples = []
    for idx, line in enumerate(open(dataset_file)):
        src_query, tgt_code = line.strip().split('\t')
        print("tgt_code:",tgt_code)
        src_query_tokens = src_query.split(' ')

        lf = parse_lambda_expr(tgt_code)
        assert lf.to_string() == tgt_code

        if reorder_predicates:
            ordered_lf = get_canonical_order_of_logical_form(lf, order_by='alphabet')
            assert ordered_lf == lf
            lf = ordered_lf

        gold_source = lf.to_string()
        grammar = transition_system.grammar
        tgt_ast = logical_form_to_ast(grammar, lf)
        reconstructed_lf = ast_to_logical_form(tgt_ast)
        assert lf == reconstructed_lf

        tgt_actions = transition_system.get_actions(tgt_ast,level_traversal=level_traversal,right_left=right_left)
        tgt_action_infos = get_action_infos(src_query_tokens, tgt_actions,level_traversal=level_traversal,right_left=right_left)
        
        #tgt_actions_teacher = transition_system.get_actions(tgt_ast,level_traversal=(not level_traversal))
        #tgt_action_infos_teacher = get_action_infos(src_query_tokens, tgt_actions_teacher,level_traversal=(not level_traversal))
        
        #print("data1:",len(tgt_action_infos))
        #print("data2:",len(tgt_action_infos_teacher))
        #print("data1:",tgt_action_infos)
        #print("data2:",tgt_action_infos_teacher)

        example = Example(idx=idx,
              src_sent=src_query_tokens,
              tgt_actions=tgt_action_infos,
              tgt_code=gold_source,
              tgt_ast=tgt_ast,
              meta=None)
        print("id:",idx)
        print("src_sent:",src_query_tokens)
        print("tgt_actions:",tgt_action_infos)
        print("tgt_code:",gold_source)
        print("tgt_ast:",tgt_ast)
        print("**********************************************")
        examples.append(example)

    return examples


def prepare_atis_dataset():
    vocab_freq_cutoff = 2
    grammar = ASDLGrammar.from_text(open('asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    transition_system = LambdaCalculusTransitionSystem(grammar)

    train_set = load_dataset(transition_system, 'data/atis/train.txt')
    dev_set = load_dataset(transition_system, 'data/atis/dev.txt')
    test_set = load_dataset(transition_system, 'data/atis/test.txt')

    # generate vocabulary
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)

    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_set]

    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=0)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_set]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=2)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, dev_set, test_set)]
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    file_name = ""
    if level_traversal:
        file_name = 'data/atis/train_teacher.bin'
    elif right_left:
        file_name = 'data/atis/train_r2l.bin'
    else:
        file_name = 'data/atis/train_student.bin'
    pickle.dump(train_set, open(file_name , 'wb'))
    pickle.dump(dev_set, open('data/atis/dev.bin', 'wb'))
    pickle.dump(test_set, open('data/atis/test.bin', 'wb'))
    pickle.dump(vocab, open('data/atis/vocab.freq%d.bin' % vocab_freq_cutoff, 'wb'))


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

    
    grammar = ASDLGrammar.from_text(open('asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    transition_system = LambdaCalculusTransitionSystem(grammar)
    # load_dataset(transition_system, 'data/atis/train.txt')
    prepare_atis_dataset()
    #global actionKind
    #actionKind = pd.DataFrame(list(set(actionKind)))
    #actionKind.to_csv("asdl/lang/py3/py3_hs_without_args.txt",index=False,header=None)

if __name__ == '__main__':
    app.run(main)
    
    # prepare_atis_dataset()
    # generate_vocab_for_paraphrase_model('data/atis/vocab.freq2.bin', 'data/atis/vocab.para.freq2.bin')
