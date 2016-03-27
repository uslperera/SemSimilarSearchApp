#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"

import multiprocessing
from semsimilar.model.document import Document
import logging
from semsimilar.exceptions import InvalidProcessorCount

ID_KEY = "Id"
TITLE_KEY = "Title"
DESCRIPTION_KEY = "Body"
TAGS_KEY = "Tags"

lock = multiprocessing.Lock()

spec = ['p', '&#xa', '&#xd', 'pre', 'code', 'blockquote', 'strong', 'ul', 'li', 'a', 'href', 'em']


def append_documents(documents, texts, final_documents, final_texts):
    """Add documents to the array"""
    logger = logging.getLogger(__name__)
    logger.debug("Acquire lock")
    lock.acquire()
    final_documents.extend(documents)
    final_texts.extend(texts)
    lock.release()
    logger.debug("Release lock")


def worker(posts, final_documents, final_texts):
    """Worker process to process documents"""
    logger = logging.getLogger(__name__)
    logger.info("Processing documents started")
    documents = []
    for post in posts:
        d = Document(post[ID_KEY], post[TITLE_KEY], post[DESCRIPTION_KEY], post[TAGS_KEY])
        d.remove_special_words(spec)
        documents.append(d)
        # documents.append(Document(post[ID_KEY], post[TITLE_KEY], post[DESCRIPTION_KEY], post[TAGS_KEY]))

    texts = []
    for doc in documents:
        texts.append(" ".join(doc.stemmed_tokens))
    logger.info("Processing documents finished")
    append_documents(documents, texts, final_documents, final_texts)
    return


def parallel_process(posts, processors):
    """Process large set of documents using parallel processing

    :param posts: posts, articles
    :param processors: number of processors
    :type posts: list<key-value object>
    :type processors: int
    :returns: Processed documents
    :rtype: list<semsimilar.semsimilar.model.document.Document>

    .. note:: Keys can be initialized before calling this function. (ID_KEY, TITLE_KEY, DESCRIPTION_KEY, TAGS_KEY)

    :Example:

    >>> with open('articles.json') as articles_file:
            articles = json.loads(articles_file.read())
    >>> parallel_process(articles, 2)
    """
    logger = logging.getLogger(__name__)
    logger.info("Parallel processing of documents started")
    if 1 > processors or multiprocessing.cpu_count() < processors:
        raise InvalidProcessorCount("Processor count " + str(processors) + " is invalid")
    else:
        corpus_size = len(posts)
        jobs = []

        manager = multiprocessing.Manager()
        final_documents = manager.list()
        final_texts = manager.list()
        logger.debug("Number of processors %s", processors)
        logger.info("Splitting of corpus started. Corpus size is %s", len(posts))
        for i in range(processors):
            if i == processors - 1:
                temp_posts = posts[
                             (corpus_size / processors) * i:(i + 1) * (
                                 corpus_size / processors) + corpus_size % processors]
            else:
                temp_posts = posts[(corpus_size / processors) * i:(i + 1) * (corpus_size / processors)]
            logger.debug("Length of the sub-corpus %s", len(temp_posts))
            p = multiprocessing.Process(target=worker, args=(temp_posts, final_documents, final_texts))
            jobs.append(p)
            p.start()

        for job in jobs:
            job.join()

        result = (final_documents, final_texts)
        return result
