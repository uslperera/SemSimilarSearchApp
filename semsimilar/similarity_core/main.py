#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"

from semsimilar.similarity_core.knowledge import lesk as lesk
import logging


def ss_similarity(documents, new_document, hal_model, count):
    """Find documents using SemSimilar similarity.

    Both HAL and Lesk based similarity calculations are used to find the most related documents.

    :param documents: documents list
    :param new_document: document to search
    :param hal_model: HAL model created from existing documents
    :param count: number of results wanted
    :type documents: list<semsimilar.semsimilar.model.document.Document>
    :type new_document: semsimilar.semsimilar.model.document.Document
    :type hal_model: semsimilar.semsimilar.similarity_core.corpus.hal.Hal
    :type count: int
    :returns: Top matched documents with their scores (0-1)
    :rtype: list<(semsimilar.semsimilar.model.document.Document, float)>

    :Example:

    >>> doc = Document(101, "PHP Session Security",
        "What are some guidelines for maintaining
        responsible session security with PHP",
        "<security><php>")
    >>> ss_similarity(documents, doc, hal, 1)
    [(document, 0.708)]
    """

    logger = logging.getLogger(__name__)
    logger.info("ss_similarity started")
    results_topic = hal_model.semantic_search(new_document.stemmed_tokens)
    logger.debug("Retrieved results from hal")
    results_ontology = []
    if results_topic:
        topic_document_ids, scores = zip(*results_topic)

        topic_documents = []
        for topic_document_id in topic_document_ids:
            topic_documents.append(documents[topic_document_id])

        results_ontology = lesk.similarity(documents=topic_documents, new_document=new_document, count=count)
        logger.debug("Retrieved results from lesk")
    return results_ontology
