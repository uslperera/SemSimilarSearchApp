#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"

import logging
from nltk.wsd import lesk


def get_synsets(tokens, window):
    """Get synsets for the tokens passed

    :param tokens: list of words
    :param window: size of the window
    :type tokens: list<string>
    :type window: int
    :returns: list of synsets
    :rtype: list<string>

    :Example:

    >>> tokens = ['this', 'is', 'a', 'demo']
    >>> get_synsets(tokens, 2)
    """
    logger = logging.getLogger(__name__)
    logger.info("Retrieving synsets started")
    window = validate_window(window)
    synsets = []
    for token in tokens:
        sentence = generate_window(window, tokens, token)
        synset = lesk(sentence, token)
        if synset is not None:
            synsets.append(synset.name())
        else:
            synsets.append(None)
    logger.debug("Tokens - %s Synsets - %s", tokens, synsets)
    logger.info("Retrieving synsets finished")
    return synsets


def generate_window(window, tokens, target):
    """Generate window to disambiguate words"""
    logger = logging.getLogger(__name__)
    logger.info("Generating window")
    new_tokens = []
    index = tokens.index(target)
    right = 0
    if len(tokens) < window + 1:
        return tokens
    # if index of the target word is greater than or equal to half of the windows size
    if index >= (window / 2):
        left = index - (window / 2)
    else:
        left = 0
        right = (window / 2) - index
    # if index of the target
    if (index + right + (window / 2)) < len(tokens):
        right = index + right + (window / 2) + 1
    else:
        right = len(tokens)
        left -= (index + (window / 2) + 1) - len(tokens)
    for num in range(left, right):
        new_tokens.append(tokens[num])
    logger.debug("Window of token %s is %s", target, new_tokens)
    logger.info("Window generated")
    return new_tokens


def validate_window(window):
    """Validate window size"""
    default_window = 4
    if window > 1 & window % 2 == 0:
        return window
    else:
        return default_window
