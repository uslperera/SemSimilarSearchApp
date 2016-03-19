from nltk.tokenize.api import TokenizerI
import re


class CodeTokenizer(TokenizerI):
    _expression = r"([?!:;\-\(\)\[\]\"/,<>]|(\.\B)|(\s'))"

    def remove_punctuations(self, s):
        return re.sub(self._expression, " ", s).strip()

    def tokenize(self, s):
        s = self.remove_punctuations(s)
        return re.split("\s+", s)

    def span_tokenize(self, s):
        pass
