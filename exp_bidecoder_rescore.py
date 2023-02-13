# coding=utf-8
from __future__ import print_function

import time

import astor
import six.moves.cPickle as pickle
from six.moves import input
from six.moves import xrange as range
from torch.autograd import Variable

import evaluation
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from common.utils import update_args, init_arg_parser

from components.dataset import Dataset
from components.reranker import *
from components.standalone_parser import StandaloneParser
from model import nn_utils
from model.paraphrase import ParaphraseIdentificationModel
from model.parser import Parser
from model.reconstruction_model import Reconstructor
from model.utils import GloveHelper

#assert astor.__version__ == "0.7.1"
if six.PY3:
    # import additional packages for wikisql dataset (works only under Python 3)
    pass


def init_config():
    
    args = arg_parser.parse_args()
    print(args)
    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def judgeParent(data1,data2,list1,list2):
    if data1.parent_t == -1 and data2.parent_t==-1:
        return True
    elif data1.parent_t == -1 or data2.parent_t == -1:
        return False
    elif str(list1[data1.parent_t].action) != str(list2[data2.parent_t].action) or  str(list1[data1.parent_t].frontier_field) !=  str(list2[data2.parent_t].frontier_field):
        return False
    else:
        return judgeParent(list1[data1.parent_t],list2[data2.parent_t],list1,list2)

def search(data1,data2,maxLen):
    dataList = []
    for item in data1:
        for i in range(len(data2)):
            if i not in dataList and str(item.action).strip() == str(data2[i].action).strip():
                if judgeParent(item,data2[i],data1,data2) : 
                    dataList.append(i)
                    break


    dataList.extend(list(range(len(data1),maxLen)))
    return dataList


def train(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    train_student_set = Dataset.from_bin_file(args.train_student_file)
    train_teacher_set = Dataset.from_bin_file(args.train_teacher_file)
        
    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
        
    else: dev_set = Dataset(examples=[])
    if args.test_file:
        test_set = Dataset.from_bin_file(args.test_file)

    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = Registrable.by_name(args.transition_system)(grammar)

    parser_cls_student = Registrable.by_name(args.parser_student)  # TODO: add arg
    parser_cls_teacher = Registrable.by_name(args.parser_teacher)  # TODO: add arg
    
    if args.pretrain:
        print('Finetune with: ', args.pretrain, file=sys.stderr)
        model_student = parser_cls_student.load(model_path=args.pretrain_student, cuda=args.cuda)
        model_teacher = parser_cls_teacher.load(model_path=args.pretrain_teacher, cuda=args.cuda)
    else:
        model_student = parser_cls_student(args, vocab, transition_system)
        model_student.level_traversal = False
        model_student.right_left = False
        model_teacher = parser_cls_teacher(args, vocab, transition_system)
        model_teacher.level_traversal = False
        model_teacher.right_left = True

    model_student.train()
    model_teacher.train()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    if args.cuda: 
        model_student.cuda()
        model_teacher.cuda()
    optimizer_cls_student = eval('torch.optim.%s' % args.optimizer_student)  # FIXME: this is evil!
    optimizer_cls_teacher = eval('torch.optim.%s' % args.optimizer_teacher)  # FIXME: this is evil!
    
    optimizer_teacher = optimizer_cls_teacher(model_teacher.parameters(), lr=args.lr)
    optimizer_student = optimizer_cls_student(model_student.parameters(), lr=args.lr)

    if not args.pretrain:
        if args.uniform_init:
            print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
            nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model_student.parameters())
            nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model_teacher.parameters())
        elif args.glorot_init:
            print('use glorot initialization', file=sys.stderr)
            nn_utils.glorot_init(model_student.parameters())
            nn_utils.glorot_init(model_teacher.parameters())

    print('begin training, %d training (teacher) examples, %d training (student) examples, %d dev examples' % (len(train_teacher_set), len(train_student_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_student_loss = report_teacher_loss = report_student_examples = \
    report_teacher_examples = report_sup_att_student_loss = report_sup_att_teacher_loss = 0.0
    history_dev_scores = []
    num_trial = patience = 0
    top_k = 1
    f = open(args.save_to+".log", 'w')
    f.close()
    while True:
        
        epoch += 1
        epoch_begin = time.time()
        index_arr = train_student_set.shuffle_data()
        for (batch_student_examples,batch_teacher_examples) in zip(train_student_set.batch_iter(batch_size=args.batch_size, index_arr=index_arr), train_teacher_set.batch_iter(batch_size=args.batch_size, index_arr=index_arr)):
            batch_student_examples = [e for e in batch_student_examples if len(e.tgt_actions) <= args.decode_max_time_step]
            batch_teacher_examples = [e for e in batch_teacher_examples if len(e.tgt_actions) <= args.decode_max_time_step]
            
            
            train_iter += 1
            optimizer_student.zero_grad()
            optimizer_teacher.zero_grad()
            
            
            ret_teacher_val = model_teacher.score(batch_teacher_examples)
            loss_teacher = -ret_teacher_val[0]

            ret_student_val = model_student.score(batch_student_examples)
            loss_student = -ret_student_val[0]
           
            
            loss_teacher_val = torch.sum(loss_teacher).data.item()
            report_teacher_loss += loss_teacher_val
            report_teacher_examples += len(batch_teacher_examples)
            loss_teacher = torch.mean(loss_teacher)
            
            loss_student_val = torch.sum(loss_student).data.item()
            report_student_loss += loss_student_val
            report_student_examples +=len(batch_student_examples)
            loss_student = torch.mean(loss_student)
               
            loss_student.backward(retain_graph=True)
           
            loss_teacher.backward(retain_graph=True)
            
            # clip gradient
            if args.clip_grad > 0.:
                grad_student_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), args.clip_grad)
                grad_teacher_norm = torch.nn.utils.clip_grad_norm_(model_teacher.parameters(), args.clip_grad)
            optimizer_teacher.step()
            optimizer_student.step()
            

            if train_iter % args.log_every == 0:
                log_student_str = '[Iter %d] encoder student loss=%.5f' %(train_iter, report_student_loss / report_student_examples)
                log_teacher_str = '[Iter %d] encoder teacher loss=%.5f' % (train_iter, report_teacher_loss / report_teacher_examples)
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_student_loss / report_student_examples)
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_teacher_loss / report_teacher_examples)
                    report_sup_att_student_loss = 0.
                    report_sup_att_teacher_loss = 0.

                print(log_student_str, file=sys.stderr)
                print(log_teacher_str, file=sys.stderr)
                report_student_loss = report_student_examples = 0.
                report_teacher_loss = report_teacher_examples = 0.
                
        # perform validation
        is_better = False
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0 and epoch >= args.vaildate_begin_epoch:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                
                # evaluation of the student model  
                eval_start = time.time()
                eval_results = evaluation.evaluateBiDecoder(dev_set.examples, model_student,model_teacher, evaluator, args, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]
                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (epoch, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start),
                                    file=open(args.save_to+".log","a+"))
                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (epoch, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start))
                
                is_better = history_dev_scores == [] or dev_score >= history_dev_scores[min(top_k,len( history_dev_scores))-1] or len( history_dev_scores) < top_k
                history_dev_scores.append(dev_score)
                history_dev_scores.sort(reverse=True)

                
                if args.test_file:
                    test_results = evaluation.evaluateBiDecoder(test_set.examples,model_student,model_teacher, evaluator, args, eval_top_pred_only=args.eval_top_pred_only)
                    test_score = test_results[evaluator.default_metric]
                    print('[Epoch %d] test details: %s, dev %s: %.5f ' % (epoch, test_results,
     evaluator.default_metric,test_score),file=open(args.save_to+".log","a+"))

        else:
            is_better = True
             
        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
            lr = optimizer_student.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)
            for param_group in optimizer_student.param_groups:
                param_group['lr'] = lr
                
            lr = optimizer_teacher.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)
            for param_group in optimizer_teacher.param_groups:
                param_group['lr'] = lr
        if is_better:
            patience = 0
            model_student_file = args.save_to +".student"  + '.bin'
            print('save the current student model ..',file=open(args.save_to+".log","a+"))
            model_student.save(model_student_file)
            torch.save(optimizer_student.state_dict(), args.save_to +".student"+'.optim.bin')
            model_teacher_file = args.save_to +".teacher" + '.bin'
            model_teacher.save(model_teacher_file)
            print('save the current teacher model ..',file=open(args.save_to+".log","a+"))
            torch.save(optimizer_teacher.state_dict(), args.save_to +".teacher" + '.optim.bin')

        elif patience < args.patience and epoch >= args.lr_decay_after_epoch and epoch >= args.vaildate_begin_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)
            
    
        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)       

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer_student.param_groups[0]['lr'] * args.lr_decay
            print('load previously best student model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to +".student"+ '.bin', map_location=lambda storage, loc: storage)
            model_student.load_state_dict(params['state_dict'])
            if args.cuda: model_student = model_student.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset student optimizer', file=sys.stderr)
                optimizer_student = torch.optim.Adam(model_student.parameters(), lr=lr)
            else:
                print('restore parameters of the student optimizers', file=sys.stderr)
                optimizer_student.load_state_dict(torch.load(args.save_to +".student"+ '.optim.bin'))

            # set new lr
            for param_group in optimizer_student.param_groups:
                param_group['lr'] = lr
            
            # decay lr, and restore from previously best checkpoint
            lr = optimizer_teacher.param_groups[0]['lr'] * args.lr_decay
            print('load previously best teacher model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to +".teacher"+ '.bin', map_location=lambda storage, loc: storage)
            model_teacher.load_state_dict(params['state_dict'])
            if args.cuda: model_teacher = model_teacher.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset teacher optimizer', file=sys.stderr)
                optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=lr)
            else:
                print('restore parameters of the teacher optimizers', file=sys.stderr)
                optimizer_teacher.load_state_dict(torch.load(args.save_to  +".teacher" + '.optim.bin'))

            # set new lr
            for param_group in optimizer_teacher.param_groups:
                param_group['lr'] = lr            
            
            
            # reset patience
            patience = 0
        

def test(args):
    
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model+".student.bin", map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser_student = parser_cls.load(model_path=args.load_model+".student.bin", cuda=args.cuda)
    parser_student.level_traversal = False
    parser_student.right_left = False
    parser_student.eval()
    
    parser_cls = Registrable.by_name(args.parser)
    parser_teacher = parser_cls.load(model_path=args.load_model+".teacher.bin", cuda=args.cuda)
    parser_teacher.level_traversal = False
    parser_teacher.right_left = True
    parser_teacher.eval()
    
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    
    
    
    
    
    test_data = []
    if args.is_dir: 
        for root, dirs, files in os.walk(args.test_file):
            test_data.extend(files)
        test_data = [args.test_file+"/"+i for i in test_data]
        print(test_data)
    else:
        test_data.append(args.test_file)
    
    result_dict = {}
    
    for data_file in test_data:
        test_set = Dataset.from_bin_file(data_file)
        eval_results, decode_results = evaluation.evaluateBiDecoder(test_set.examples,parser_student,
                                                                parser_teacher, evaluator, args,
                                                                verbose=args.verbose,
                                                                return_decode_result=True)
        result_dict[data_file] = eval_results
    
    print(args.load_model, file=sys.stderr)
    for i in result_dict:
        print(i,":",result_dict[i], file=sys.stderr)
        
        
        



if __name__ == '__main__':
    arg_parser = init_arg_parser() 
    args = init_config()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)