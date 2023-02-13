from components.evaluator import Evaluator
from common.registerable import Registrable
import ast
import astor

import sys, traceback
import numpy as np



@Registrable.register('ifttt_evaluator')
class IFTTTEvaluator(Evaluator):
    def __init__(self, transition_system=None, args=None):
        super(IFTTTEvaluator, self).__init__()
        self.transition_system = transition_system
        self.default_metric = 'accuracy'

    def is_hyp_correct(self, example, hyp, layer=3):
        if layer == 3:
            return self.transition_system.compare_ast(hyp.tree, example.tgt_ast)
        else:
            exe_code = example.tgt_code
            hyp_code = self.transition_system.ast_to_surface_code(hyp.tree)
            exe_code = exe_code.split(" ")
            hyp_code = hyp_code.split(" ")
            if exe_code[4] == hyp_code[4] and exe_code[10] == hyp_code[10]:
                return True
            else:
                return False


    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        correct_array = []
        oracle_array = []
        correct_2layer_array = []
        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):
                    try:
                        is_correct = self.is_hyp_correct(example, hyp)
                        is_correct2 = self.is_hyp_correct(example, hyp,layer=2)
                    except:
                        is_correct = False
                        is_correct2 = False

                        print('-' * 60, file=sys.stdout)
                        print('Error in evaluating Example %s, hyp %d {{ %s }}' % (example.idx, hyp_id, hyp.code),
                              file=sys.stdout)

                        print('example id: %s, hypothesis id: %d' % (example.idx, hyp_id), file=sys.stdout)
                        traceback.print_exc(file=sys.stdout)
                        print('-' * 60, file=sys.stdout)

                    hyp.is_correct = is_correct
                    hyp.is_correct2 = is_correct2

                correct_array.append(hyp_list[0].is_correct)
                correct_2layer_array.append(hyp_list[0].is_correct2)
                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))
            else:
                correct_array.append(False)
                correct_2layer_array.append(False)
                oracle_array.append(False)

        acc = np.average(correct_array)
        acc_channel = np.average(correct_2layer_array)

        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            acc_channel=acc_channel,
                            oracle_accuracy=oracle_acc)

        return eval_results