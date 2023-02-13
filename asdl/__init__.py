import six
from .lang.lambda_dcs.lambda_dcs_transition_system import LambdaCalculusTransitionSystem


if six.PY2:
    from .lang.py.py_transition_system import PythonTransitionSystem
else:
    from .lang.py3.py3_transition_system import Python3TransitionSystem
    from .lang.ifttt.ifttt_dcs_transition_system import IFTTTTransitionSystem