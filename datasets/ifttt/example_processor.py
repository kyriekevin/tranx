from common.registerable import Registrable
from datasets.utils import ExampleProcessor

from asdl.asdl_ast import AbstractSyntaxTree

@Registrable.register('ifttt_example_processor')
class IFTTTExampleProcessor(ExampleProcessor):
    def __init__(self, transition_system):
        self.transition_system = transition_system

    def pre_process_utterance(self, utterance):
        slot2entity_map = dict()

        return utterance, slot2entity_map

    def post_process_hypothesis(self, hyp, meta_info, utterance=None):
        """traverse the AST and replace slot ids with original strings"""
        slot2entity_map = meta_info

        def _travel(root):
                for field in root.fields:
                    if self.transition_system.grammar.is_primitive_type(field.type):
                        slot_name = field.value
                        if slot_name in slot2entity_map:
                            field.value = slot2entity_map[slot_name]
                    else:
                        for val in field.as_value_list:
                            _travel(val)

        _travel(hyp.tree)
        hyp.code = self.transition_system.ast_to_surface_code(hyp.tree)
