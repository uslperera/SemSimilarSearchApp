#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"

import numpy.linalg as LA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import logging
import collections


class HAL(object):
    """Hyperspace Analogue to Language

    Can be used to find similar documents using keywords and word co-occurrences.

    :param documents: documents list
    :type documents: list<semsimilar.semsimilar.model.document.Document>
    :returns: HAL model
    :rtype: semsimilar.semsimilar.similar.corpus.hal.HAL

    **Property**:
     - co_occurrence_matrix
     - document_term_matrix
     - threshold
     - vocabulary

    **Setter**
     - threshold

    :Example:

    >>> documents = ["first document", "second document"]
    >>> hal = HAL(documents=documents)
    [(document, 0.708)]
    """
    __tfidf = None
    __dtm = None
    __cm = None
    __threshold = 0.1
    __semantic_threshold = 0.4
    __vocabulary = None

    __logger = None

    def __init__(self, documents):
        self.__logger = logging.getLogger(__name__)
        self.__logger.info("HAL model creation started")
        self.__tfidf = TfidfVectorizer(input="content")
        self.create_document_term_matrix(documents)
        self.create_co_occurrence_matrix(documents)
        self.__logger.info("HAL model creation finished")

    @property
    def co_occurrence_matrix(self):
        return self.__cm

    @property
    def document_term_matrix(self):
        return self.__dtm

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold):
        if 0 <= threshold < 1:
            self.__threshold = threshold

    @property
    def vocabulary(self):
        return self.__vocabulary

    @staticmethod
    def cosine(a, b):
        # Find the cosine distance between two vectors
        try:
            result = round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)
        except ZeroDivisionError:
            result = 0
        return result

    def create_document_term_matrix(self, documents):
        """Create document term matrix

        :param documents: documents list
        :type documents: list<semsimilar.semsimilar.model.document.Document>
        :returns: void

        :Example:

        >>> documents = ["first document", "second document"]
        >>> hal.create_document_term_matrix(documents)
        """
        logging.info("Started creating TFidf matrix")
        self.__dtm = self.__tfidf.fit_transform(documents).toarray()
        self.__vocabulary = np.array(self.__tfidf.get_feature_names())

    def create_co_occurrence_matrix(self, documents):
        """Create term co-occurrence matrix.

        :param documents: documents list
        :type documents: list<semsimilar.semsimilar.model.document.Document>
        :returns: void

        :Example:

        >>> documents = ["first document", "second document"]
        >>> hal.create_co_occurrence_matrix(documents)
        """
        logging.info("Started creating co-occurrence matrix")
        x = self.document_term_matrix
        cooccurrence_matrix = np.dot(x.transpose(), x)
        cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
        with np.errstate(divide='ignore', invalid='ignore'):
            cooccurrence_matrix_percentage = np.true_divide(cooccurrence_matrix,
                                                            cooccurrence_matrix_diagonal[:, None])
        self.__cm = cooccurrence_matrix_percentage
        logging.info("Finished creating co-occurrence matrix")

    def convert_to_vector_space(self, query):
        """Convert text into vector space.

        :param query: list of text
        :type query: list<string>
        :returns: a matrix
        :rtype: numpy.ndarray

        :Example:

        >>> documents = ["first document", "second document"]
        >>> hal = HAL(documents=documents)
        >>> new_document = Document(id=1, title="new document",
            None, None)
        >>> qtm = hal.convert_to_vector_space(new_document.stemmed_tokens)
        [[0 1 0 1 1]]
        _
        """
        logging.info("Started converting the text to vector space")
        vectorizer = TfidfVectorizer(input="content", vocabulary=self.__tfidf.get_feature_names())
        query_string = " ".join(query)
        logging.debug("Query string %s", query_string)
        vector = vectorizer.fit_transform([query_string]).toarray()
        return vector

    def semantic_search(self, query):
        """Search for a document semantically.

        Uses both co-occurrence and keyword search functions.

        :param query: list of text
        :type query: list<string>
        :returns: list of document ids and scores
        :rtype: list<(int, float)>

        :Example:

        >>> documents = ["first document", "second document"]
        >>> hal = HAL(documents=documents)
        >>> new_document = Document(id=1, title="new document",
            None, None)
        >>> qtm = hal.convert_to_vector_space(new_document.stemmed_tokens)
        >>> hal.semantic_search(new_document.stemmed_tokens, qtm)
        [(1, 0.708)]
        _
        """
        logging.info("Search semantically")
        qtm = self.convert_to_vector_space(query)
        results1 = self.keyword_search(query, qtm)
        results2 = self.co_occurrence_search(query, qtm)

        results = results1 + results2

        clusterer = collections.defaultdict(list)
        for l in results:
            k, v = l
            clusterer[k].append(v)

        final_results = clusterer.items()

        final_results.sort(key=lambda tup: tup[1][0], reverse=True)
        logging.debug("Semantic search result %s", final_results)
        return final_results

    def co_occurrence_search(self, query, qtm):
        """Search for a document using co-occurrence of words.

        :param query: list of text
        :param qtm: query in vector space
        :type query: list<string>
        :type qtm: numpy.ndarray
        :returns: list of document ids and scores
        :rtype: list<(int, float)>

        :Example:

        >>> documents = ["first document", "second document"]
        >>> hal = HAL(documents=documents)
        >>> new_document = Document(id=1, title="new document",
            None, None)
        >>> qtm = hal.convert_to_vector_space(new_document.stemmed_tokens)
        >>> hal.co_occurrence_search(new_document.stemmed_tokens, qtm)
        [(1, 0.708)]
        _
        """
        logging.info("Co-occurrence search")
        semantic_term_ids = set(self.get_related_vocabulary(query))
        doc_ids = []
        for term_id in semantic_term_ids:
            docs = np.where(self.__dtm[:, term_id] != 0)[0]
            doc_ids.extend(docs)

        results = []
        for id in set(doc_ids):
            cos = self.cosine(qtm, self.__dtm[id])
            if cos > 0:
                doc = (id, cos)
                results.append(doc)

        results.sort(key=lambda tup: tup[1], reverse=True)
        return results[:10]

    '''
    def temp_search(self, query, qtm):
        term_ids = []
        for term in query:
            term_id = self.get_term_id(term)
            if term_id is not None:
                term_ids.append(term_id)

        doc_ids = []
        for term_id in term_ids:
            doc_ids.extend(np.where(self.__dtm[:, term_id] != 0)[0])

        results = []
        for id in set(doc_ids):
            #Rem-----
            t_ids = np.where(self.__dtm[id, :] != 0)[0]
            # t_ids = self.__dtm[id, :]
            q_ids = np.where(qtm[0, :] != 0)[0]
            # q_ids = qtm[:, :]
            aa = np.intersect1d(t_ids, q_ids)

            score = 0
            for i in aa:
                c = self.__dtm[id, i] + qtm[:, i]
                score += np.log10(c)[0] + 1

            doc = (id, score)
            results.append(doc)

        results.sort(key=lambda tup: tup[1], reverse=True)
        return results[:10]
        #-----
    '''
    def keyword_search(self, query, qtm):
        """Search for a document using keywords.

        :param query: list of text
        :param qtm: query in vector space
        :type query: list<string>
        :type qtm: numpy.ndarray
        :returns: list of document ids and scores
        :rtype: list<(int, float)>

        :Example:

        >>> documents = ["first document", "second document"]
        >>> hal = HAL(documents=documents)
        >>> new_document = Document(id=1, title="new document",
            None, None)
        >>> qtm = hal.convert_to_vector_space(new_document.stemmed_tokens)
        >>> hal.keyword_search(new_document.stemmed_tokens, qtm)
        [(1, 0.708)]
        _
        """
        logging.info("Keywords search")
        term_ids = []
        for term in query:
            term_id = self.get_term_id(term)
            if term_id is not None:
                term_ids.append(term_id)

        doc_ids = []
        for term_id in term_ids:
            doc_ids.extend(np.where(self.__dtm[:, term_id] != 0)[0])

        results = []
        for id in set(doc_ids):
            cos = self.cosine(qtm, self.__dtm[id])
            if cos > self.__threshold:
                doc = (id, cos)
                results.append(doc)

        results.sort(key=lambda tup: tup[1], reverse=True)
        return results[:10]

    def get_related_vocabulary(self, query):
        # Get co-occurring terms in the vocabulary
        logging.info("Started getting related vocabulary")
        word_ids = []
        for term in query:
            term_id = np.where(self.__vocabulary == term)[0]
            if len(term_id) > 0:
                id = term_id[0]
                for i, score in enumerate(self.__cm[id, :]):
                    if score > self.__semantic_threshold:
                        word_ids.append(i)
        return word_ids

    def get_term_id(self, term):
        """Get id of the term in vocabulary.

        :param term: word
        :type term: string
        :returns: index of the term in array
        :rtype: int

        :Example:

        >>> hal.get_term_id('apple')
        21
        """
        logging.info("Get term id")
        term_id = np.where(self.__vocabulary == term)[0]
        if len(term_id) > 0:
            return term_id[0]
        return None
