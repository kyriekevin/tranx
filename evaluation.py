# coding=utf-8
from __future__ import print_function

import sys
import traceback
from tqdm import tqdm
from components.action_info import ActionInfo, get_action_infos
from copy import deepcopy

def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    is_wikisql = args.parser == 'wikisql_parser'

    decode_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if is_wikisql:
            hyps = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size)
        else:
            hyps = model.parse(example.src_sent, context=None, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                
                got_code = True
                
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
            ' '.join(example.src_sent), example.tgt_code,hyp_id,hyp.tree.to_string()),file=sys.stdout)
                    if got_code:                        
                        print(hyp.code)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1
        
        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results


def decodeBiDecoder(examples, model, model_rescore, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()
    model_rescore.eval()

    is_wikisql = args.parser == 'wikisql_parser'

    decode_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if is_wikisql:
            hyps = model.parse(example.src_sent, context=example.table, beam_size=int(args.beam_size/2))
            hyps_back = model_rescore.parse(example.src_sent,context=example.table,
                                            beam_size=int(args.beam_size/2))
            
        else:
            hyps = model.parse(example.src_sent, context=None, beam_size=args.beam_size)
            hyps_back = model_rescore.parse(example.src_sent,context=None,
                                            beam_size=int(args.beam_size/2))
            
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                newExample = deepcopy(example)
                tgt_actions = model.transition_system.get_actions(hyp.tree,level_traversal=model.level_traversal,right_left= not model.right_left)
                newExample.tgt_actions = get_action_infos(example.src_sent, tgt_actions,level_traversal=model.level_traversal,right_left= not model.right_left)
                newscore = model_rescore.score([newExample])
                hyp.score += newscore[0].data.item()
                
                got_code = True
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    traceback.print_exc(file=sys.stdout)

        for hyp_id, hyp in enumerate(hyps_back):
            got_code = False
            try:
                hyp.code = model_rescore.transition_system.ast_to_surface_code(hyp.tree)
                newExample = deepcopy(example)
                tgt_actions = model_rescore.transition_system.get_actions(hyp.tree,
                                            level_traversal=model_rescore.level_traversal,right_left= not model.right_left)
                newExample.tgt_actions = get_action_infos(example.src_sent,
                                        tgt_actions,level_traversal=model_rescore.level_traversal, right_left= not model.right_left)
                newscore = model.score([newExample])
                hyp.score += newscore[0].data.item()
                
                got_code = True
                decoded_hyps.append(hyp)
                
            except:
                if verbose:
                    traceback.print_exc(file=sys.stdout)
        

        count += 1
        decoded_hyps.sort(key=lambda hyp: -hyp.score)
        decode_results.append(decoded_hyps)

    if was_training: model.train()
    if was_training: model_rescore.train()

    return decode_results



def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
   
    decode_results = decode(examples, parser, args, verbose=verbose)
    
    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only,)
    
    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
    
def evaluateBiDecoder(examples, parser1, parser2,evaluator,
                      args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    decode_results = decodeBiDecoder(examples, parser1, parser2, args, verbose=False)
    
    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only,)
    
    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result



