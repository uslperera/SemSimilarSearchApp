import numpy.linalg as LA
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import *
import numpy as np

class HAL(object):
    __tfidf = None
    __dtm = None
    __wwm = None
    __threshold = 0.1
    __semantic_thresold = 0.4
    __vocabulary = None

    @property
    def word_word_matrix(self):
        return self.__wwm

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

    def __init__(self, documents):
        self.__tfidf = TfidfVectorizer(input="content")
        self.__dtm = self.__tfidf.fit_transform(documents)
        self.__vocabulary = np.array(self.__tfidf.get_feature_names())
        self.create_word_to_word_matrix(documents)

    @staticmethod
    def cosine(a, b):
        if (LA.norm(a) * LA.norm(b)) != 0:
            return round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)
        return 0

    def create_word_to_word_matrix(self, documents):
        l = len(self.__vocabulary)
        self.__wwm = np.zeros((l, l), dtype=np.float)

        for doc in documents:
            tokens = doc.split(" ")
            for f_token in tokens:
                term_id = self.get_term_id(f_token)
                if term_id is None:
                    continue
                for token in tokens:
                    x = self.get_term_id(token)
                    if x is not None:
                        self.__wwm[term_id, x] = 1

        wwm = np.zeros((l, l), dtype=np.float)
        for y, v in enumerate(self.__vocabulary):
            for x, v in enumerate(self.__vocabulary):
                score = self.cosine(self.__wwm[:, x], self.__wwm[:, y])
                wwm[x, y] = score
        self.__wwm = wwm

    def add_document(self, document):
        count = CountVectorizer(input="content", stop_words="english", vocabulary=self.__vocabulary)
        dtm1 = count.fit_transform([" ".join(document)])

        dtm2 = np.append(self.__dtm.toarray(), dtm1.toarray(), axis=0)
        self.__dtm = csr_matrix(dtm2)

    def get_related_vocabulary(self, query):
        word_ids = []
        for term in query:
            term_id = np.where(self.__vocabulary == term)[0]
            if len(term_id) > 0:
                id = term_id[0]
                for i, score in enumerate(self.__wwm[id, :]):
                    if score > self.__semantic_thresold:
                        word_ids.append(i)
        return word_ids

    def search(self, query):
        countVectorizer = CountVectorizer(input="content", stop_words="english",
                                          vocabulary=self.__tfidf.get_feature_names())
        query_string = " ".join(query)
        qtm = countVectorizer.fit_transform([query_string]).toarray()

        dtm = self.__dtm.toarray()

        term_ids = []
        for term in query:
            term_id = self.get_term_id(term)
            if term_id is not None:
                term_ids.append(term_id)

        similar_docs = []
        for term_id in term_ids:
            similar_docs.extend(np.where(dtm[:, term_id] != 0)[0])

        similar = set(similar_docs)

        results = []
        for s in similar:
            cos = self.cosine(qtm, dtm[s])
            if cos > self.__threshold:
                doc = (s, cos)
                results.append(doc)

        """"""
        semantic_term_ids = set(self.get_related_vocabulary(query))
        se_doc_ids = []
        for term_id in semantic_term_ids:
            doc = np.where(dtm[:, term_id] != 0)[0][0]
            se_doc_ids.append(doc)

        for s in (set(se_doc_ids) - similar):
            doc = (s, 0)
            results.append(doc)
        """"""

        results.sort(key=lambda tup: tup[1], reverse=True)
        return results

    def get_term_id(self, term):
        term_id = np.where(self.__vocabulary == term)[0]
        if len(term_id) > 0:
            return term_id[0]
        return None
