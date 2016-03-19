from nltk.corpus import wordnet as wn
from nltk.metrics import distance


def similarity(documents, new_document, count):
    count = __validate_count(count)

    results = []
    for document in documents:
        score = __get_score(new_document, document)
        if len(results) < count:
            result = (document, score)
            results.append(result)
            if len(results) == count:
                results.sort(key=lambda tup: tup[1], reverse=True)
        elif results[count - 1][1] < score:
            results.pop()
            result = (document, score)
            results.append(result)
            results.sort(key=lambda tup: tup[1], reverse=True)
    return results


def __validate_count(count):
    default_count = 1
    if count > 0:
        return count
    else:
        return default_count


def __get_score(new_doc, doc):
    if new_doc.synsets is None or doc.synsets is None:
        return 0
    # total1 = __calculate_semantic_score(new_doc.synsets, doc.synsets)
    total1 = __calculate_semantic_score(new_doc.synsets, doc.synsets) + __calculate_string_score(new_doc.synsets,
                                                                                                 new_doc.synset_tokens,
                                                                                                 doc.synset_tokens)
    # total2 = __calculate_semantic_score(doc.synsets, new_doc.synsets)

    total2 = __calculate_semantic_score(doc.synsets, new_doc.synsets) + __calculate_string_score(doc.synsets,
                                                                                                 doc.synset_tokens,
                                                                                                 new_doc.synset_tokens)

    return (total1 + total2) / (len(doc.synset_tokens) + len(new_doc.synset_tokens))


def __calculate_string_score(synsets, tokens1, tokens2):
    total = 0
    for index, syn in enumerate(synsets, start=0):
        max = 0
        if syn is None:
            for token2 in tokens2:
                sim = 1 - distance.jaccard_distance(set(tokens1[index]), set(token2))
                if sim is not None and sim > max:
                    max = sim
        total += max
    return total


def __calculate_semantic_score(synsets1, synsets2):
    total = 0
    for syn1 in synsets1:
        max = 0
        if syn1 is not None:
            for syn2 in synsets2:
                if syn2 is not None:
                    sim = wn.synset(syn1).path_similarity(wn.synset(syn2))
                    if sim is not None and sim > max:
                        max = sim
        total += max
    return total
