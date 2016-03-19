from nltk.wsd import lesk


def get_synsets(tokens, window):
    window = validate_window(window)
    synsets = []
    for token in tokens:
        sentence = generate_window(window, tokens, token)
        synset = lesk(sentence, token)
        if synset is not None:
            synsets.append(synset.name())
        else:
            synsets.append(None)
    return synsets


def generate_window(window, tokens, target):
    new_tokens = []
    index = tokens.index(target)
    right = 0
    if len(tokens) < window + 1:
        return tokens
    # if index of the target word is greater than or equal to half of the windows size
    if index >= (window / 2):
        left = index - (window / 2)
    else:
        left = 0
        right = (window / 2) - index
    # if index of the target
    if (index + right + (window / 2)) < len(tokens):
        right = index + right + (window / 2) + 1
    else:
        right = len(tokens)
        left -= (index + (window / 2) + 1) - len(tokens)
    for num in range(left, right):
        new_tokens.append(tokens[num])

    return new_tokens


def validate_window(window):
    default_window = 4
    if window > 1 & window % 2 == 0:
        return window
    else:
        return default_window
