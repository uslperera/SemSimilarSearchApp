#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"

from nltk.tokenize.api import TokenizerI
from semsimilar.core.textprocessor.replacers import RegexpReplacer
import re


class CodeTokenizer(TokenizerI):
    _expression = r"([?!:;\-\(\)\[\]\"/,<>]|(\.\B)|(\s'))"
    __replacer = RegexpReplacer()

    def remove_punctuations(self, s):
        """Remove punctuation marks"""
        return re.sub(self._expression, " ", s).strip()

    def tokenize(self, s):
        """Tokenize a string (Splits the text into words)

        :param s: stream of text
        :type posts: string
        :returns: list of words
        :rtype: list<string>

        :Example:

        >>> c = CodeTokenizer()
        >>> c.tokenize("Stream of text 123")
        """
        s = s.lower()
        s = self.remove_punctuations(s)
        s = self.__replacer.replace(s)
        return re.split("\s+", s)

    def span_tokenize(self, s):
        pass
