from gensim import similarities


def similarity(lda_model, dictionary, corpus, documents, new_document, count):
    count = __validate_count(count)
    vec_bow = dictionary.doc2bow(new_document.get_stemmed_tokens())
    # convert the query to LDA space
    vec_lda = lda_model[vec_bow]

    index = similarities.MatrixSimilarity(lda_model[corpus])
    index.num_best = count
    sims = index[vec_lda]

    results = []
    for sim in sims:
        score = sim[1]
        document = documents[sim[0]]
        result = (document, score)
        results.append(result)

    return results


def __validate_count(count):
    default_count = 1
    if count > 0:
        return count
    else:
        return default_count
