from nltk.tokenize.api import TokenizerI
from semsimilar.core.textprocessor.wsd import get_synsets
from semsimilar.core.textprocessor.processor import *


class Document(object):
    __id = None
    __title = None
    __description = None
    __tags = None
    __tokens = None
    __synsets = None
    __tokenizer = None
    __window = 4
    title_enabled = True
    description_enabled = tags_enabled = False
    __synset_tokens = None

    def __init__(self, id, title, description, tags):
        self.__id = id
        self.__title = title
        self.__description = description
        self.__tags = tags
        self.generate_tokens()

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title):
        self.__title = title

    @property
    def description(self):
        return self.__description

    @description.setter
    def description(self, description):
        self.__description = description

    @property
    def tags(self):
        return self.__tags

    @tags.setter
    def tags(self, tags):
        self.__tags = tags

    @property
    def tokens(self):
        return self.__tokens

    @property
    def synset_tokens(self):
        return self.__synset_tokens

    @property
    def synsets(self):
        return self.__synsets

    @staticmethod
    def set_window(window):
        if window > 1 & window % 2 == 0:
            Document.__window = window

    @staticmethod
    def set_tokenizer(tokenizer):
        if isinstance(tokenizer, TokenizerI):
            Document.__tokenizer = tokenizer

    def generate_tokens(self):
        self.__tokens = []
        if self.title_enabled & self.description_enabled & self.tags_enabled:
            text = self.__title + " " + self.__description + " " + self.__tags
            self.__synset_tokens = self.__tokenizer.tokenize((self.__title + " " + self.__tags).lower())
        elif self.title_enabled & self.description_enabled:
            text = self.__title + " " + self.__description
            self.__synset_tokens = self.__tokenizer.tokenize(self.__title.lower())
        elif self.title_enabled & self.tags_enabled:
            text = self.__title + " " + self.__tags
            self.__synset_tokens = self.__tokenizer.tokenize((self.__title + " " + self.__tags).lower())
        else:
            text = self.__title
            self.__synset_tokens = self.__tokenizer.tokenize(self.__title.lower())

        tokens = self.__tokenizer.tokenize(text.lower())
        tokens = remove_stopwords(tokens)
        self.__tokens = tokens

        self.__synset_tokens = remove_stopwords(self.__synset_tokens)
        # self.__synsets = get_synsets(self.__tokens, self.__window)
        self.__synsets = get_synsets(self.__synset_tokens, self.__window)

    def get_stemmed_tokens(self):
        return stem_tokens(self.__tokens)
