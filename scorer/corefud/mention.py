from scorer.base.mention import Mention


class CorefUDMention(Mention):
    """Representation of (potentially non-contiguous) mention for the evaluation script.
    It must allow mention matching in two documents that are aligned, yet likely marked with different coreference relations.
    A mention is thus defined only by position of the words (sentence ord and word ord) the mention is formed by.

    As mentions are allowed to be non-contiguous, matching of such mentions must be supported. Matching only the start and
    the end of the mentions is insufficient in this case. Matching in this class is therefore based on matching the sets
    of words that form the mentions.

    The class allows for both exact and partial matching. This is controlled by the head of the mention which can be specified
    as well. If head is defined for at least one of the mentions in comparison, the mentions can be matched by partial/fuzzy
    matching. If none of the two mentions have a head specified, the mentions are compared using the exact matching.
    """

    class WordOrd:
        """Representation of a mention word for evaluation purposes.
        A word is defined only by its position within the document, i.e. ordinal number of the word within a sentence and the
        sentence within the document. For this reason, comaprison operators are defined for the class.
        """

        def __init__(self, node):
            self._sentord = node.root.bundle.number
            self._wordord = node.ord

        def __lt__(self, other):
            if isinstance(other, self.__class__):
                if self._sentord == other._sentord:
                    return self._wordord < other._wordord
                else:
                    return self._sentord < other._sentord
            return NotImplemented

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                if self._sentord == other._sentord \
                    and self._wordord == other._wordord:
                    return True
                else:
                    return False
            return NotImplemented

        def __ne__(self, other):
            return not self.__eq__(other)

        def __le__(self, other):
            return self.__lt__(other) or self.__eq__(other)

        def __str__(self):
            return f"{self._sentord}-{self._wordord}"

        def __repr__(self):
            return str(self)

        def __hash__(self):
            return hash((self._sentord, self._wordord))

    def __init__(self, nodes, head=None):
        super().__init__()
        self._words = [CorefUDMention.WordOrd(n) for n in nodes]
        self._words.sort()
        self._wordsset = set(self._words)
        if head:
            self._minset.add(CorefUDMention.WordOrd(head))
            self._is_zero = head.is_empty()
        else:
            self._is_zero = nodes[0].is_empty()
