#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import stop_words
import logging
from nltk.corpus import stopwords

# en_stop = stop_words.get_stop_words('en')
en_stop = stopwords.words('english')
p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer("english")


def remove_stopwords(tokens):
    """Remove stop words in English

    :param tokens: list of words
    :type tokens: list<string>
    :returns: list of words
    :rtype: list<string>

    :Example:

    >>> tokens = ['this', 'is', 'a', 'demo']
    >>> remove_stopwords(tokens)
    """
    logger = logging.getLogger(__name__)
    logger.info("Stopwords removal started")
    stopped_tokens = [i for i in tokens if not i in en_stop]
    logger.debug("Stopped tokens %s", stopped_tokens)
    logger.info("Stopwords removal finished")
    return stopped_tokens


def stem_tokens(tokens):
    """Extract the stem of tokens

    :param tokens: list of words
    :type tokens: list<string>
    :returns: list of words
    :rtype: list<string>

    :Example:

    >>> tokens = ['this', 'is', 'a', 'demo']
    >>> stem_tokens(tokens)
    """
    logger = logging.getLogger(__name__)
    logger.info("Stemming started")
    stemmed_tokens = [s_stemmer.stem(i) for i in tokens]
    logger.debug("Stemmed tokens %s", stemmed_tokens)
    logger.info("Stemming finished")
    return stemmed_tokens


def remove_custom_words(custom_words, tokens):
    """Remove custom list of words

    :param custom_words: list of words to be removed
    :param tokens: list of words
    :type custom_words: list<string>
    :type tokens: list<string>
    :returns: list of words
    :rtype: list<string>

    :Example:

    >>> tokens = ['this', 'is', 'a', 'demo']
    >>> remove_custom_words(['is'], tokens)
    """
    logger = logging.getLogger(__name__)
    logger.info("Custom words removal started")
    filtered_tokens = [i for i in tokens if not i in custom_words]
    logger.debug("Tokens after custom words were removed %s", filtered_tokens)
    logger.info("Custom words removal finished")
    return filtered_tokens
