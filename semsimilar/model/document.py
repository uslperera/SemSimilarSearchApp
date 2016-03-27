#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"

from nltk.tokenize.api import TokenizerI
from semsimilar.textprocessor.wsd import get_synsets
from semsimilar.textprocessor.processor import *
import logging


class Document(object):
    """
    Document class includes the basic skeleton to keep all the data associated with posts or articles.

    :param id: unique number of the document
    :param title: title of the document
    :param description: description of the document
    :param tags: tags of the document
    :type id: int
    :type title: string
    :type description: string
    :type tags: string
    :returns: Document model
    :rtype: semsimilar.semsimilar.model.document.Document

    **Property**:
     - id
     - title
     - description
     - tags
     - synset_tokens (Get synsets of the tokens excluding Description)
     - stemmed_tokens
     - tokens
     - synsets (Get synsets of the tokens including Description)

    **Setter**
     - id
     - title
     - description
     - tags


    :Example:

    >>> Document(101, "PHP Session Security",
    "What are some guidelines for maintaining
    responsible session security with PHP",
     "<security><php>")
    """
    __id = None
    __title = None
    __description = None
    __tags = None
    __tokens = None
    __synsets = None
    __stemmed_tokens = None
    __tokenizer = None
    __window = 4
    title_enabled = True
    description_enabled = tags_enabled = False
    __synset_tokens = None

    __logger = None

    def __init__(self, id, title, description, tags):
        self.__logger = logging.getLogger(__name__)
        self.__logger.info("Started creating document with id %s", id)

        self.__id = id
        self.__title = title
        self.__description = description
        self.__tags = tags
        self.generate_tokens()
        self.__logger.info("Finished creating document with id %s", id)

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
    def synset_tokens(self):
        return self.__synset_tokens

    @property
    def stemmed_tokens(self):
        return self.__stemmed_tokens

    @property
    def tokens(self):
        return self.__tokens

    @property
    def synsets(self):
        return self.__synsets


    @staticmethod
    def set_window(window):
        """
        Set the size of the window to be used for word sense disambiguation

        The value must be an even number greater than 1

        :param window: size of the window
        :type window: int
        :returns: void

        :Example:

        >>> Document.set_window(4)
        """
        if window > 1 & window % 2 == 0:
            Document.__window = window

    @staticmethod
    def set_tokenizer(tokenizer):
        """
        Set a tokenizer to extract words

        .. note:: Possible tokenizers are semsimilar.semsimilar.textprocessor.tokenize.CodeTokenizer, nltk.tokenize.api.*

        :param tokenizer: tokenizer object
        :type tokenizer: semsimilar.semsimilar.textprocessor.tokenize.CodeTokenizer, nltk.tokenize.api.* ,...
        :returns: void

        :Example:

        >>> Document.set_tokenizer(CodeTokenizer())
        """
        if isinstance(tokenizer, TokenizerI):
            Document.__tokenizer = tokenizer

    def remove_special_words(self, words):
        self.__tokens = remove_custom_words(words, self.__tokens)
        self.__synsets = get_synsets(self.__synset_tokens, self.__window)
        self.__stemmed_tokens = stem_tokens(self.__tokens)

    def generate_tokens(self):
        """Tokenize the document based on selected components (title | description | tags)

        .. note:: This function is automatically called during a document creation. Can be called if content of the document is changed.

        :returns: void

        :Example:

        >>> doc = Document(101, "PHP Session Security", None, None)
        >>> doc.generate_tokens()
        """
        self.__logger.info("NLP processing started")
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
        self.__logger.debug("Tokens %s", tokens)
        self.__tokens = remove_stopwords(tokens)
        self.__logger.debug("Tokens after stop words are removed %s", self.__tokens)

        # synset_tokens = remove_stopwords(self.__synset_tokens)
        synset_tokens = remove_stopwords(self.__tokens)
        self.__synset_tokens = list(set(synset_tokens))
        self.__logger.debug("Synset tokens %s", self.__synset_tokens)
        self.__synsets = get_synsets(self.__synset_tokens, self.__window)
        self.__logger.debug("Synsets %s", self.__synsets)
        self.__stemmed_tokens = stem_tokens(self.__tokens)
        self.__logger.debug("Stemmed tokens %s", self.__stemmed_tokens)
        self.__logger.info("Finished processing the document")
