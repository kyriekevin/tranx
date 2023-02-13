# coding=utf-8

from asdl.asdl import *
from asdl.hypothesis import Hypothesis
from asdl.transition_system import *


class DecodeHypothesis(Hypothesis):
    def __init__(self,level_traversal=False,right_left=False):
        super(DecodeHypothesis, self).__init__()

        self.action_infos = []
        self.code = None
        self.level_traversal=level_traversal
        self.right_left=right_left

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action

        new_hyp = self.clone_and_apply_action(action,level_traversal=self.level_traversal,right_left=self.right_left)
        new_hyp.action_infos.append(action_info)

        return new_hyp

    def copy(self):
        new_hyp = DecodeHypothesis(self.level_traversal,self.right_left)
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.action_infos = list(self.action_infos)
        new_hyp.score = self.score
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp.t = self.t
        new_hyp.code = self.code

        new_hyp.update_frontier_info(self.level_traversal,self.right_left)

        return new_hyp
