import six

from datasets.django.evaluator import DjangoEvaluator


if six.PY3:
    from datasets.ifttt.evaluator import IFTTTEvaluator

