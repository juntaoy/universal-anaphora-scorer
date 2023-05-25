class Mention:
    """Base class for a mention.
    
    Here we only include the properties that might be used outside the mention class.
    We assign default values to them to avoid an error even if not overriden by a class
    corresponding to a specific format.
    """
    def __init__(self):
        self._words = []  # store all word indices
        self._wordsset = set()
        self._minset = set()
        self._is_referring = True  # for non-referring
        self._is_split_antecedent = False  # for split-antecedent
        self._split_antecedent_sets = set()  # for split-antecedent
        self._is_zero = False

    @property
    def words(self):
        return self._words

    @property
    def start(self):
        return self._words[0]

    @property
    def end(self):
        return self._words[-1]

    @property
    def is_zero(self):
        return self._is_zero

    @property
    def is_referring(self):
        return self._is_referring

    @property
    def is_split_antecedent(self):
        return self._is_split_antecedent

    @property
    def split_antecedent_sets(self):
        return self._split_antecedent_sets

    def __getitem__(self, i):
        return self._words[i]

    def __len__(self):
        return len(self._words)

    def __eq__(self, other):
        return self._exact_match(other)

    # the _exact_match is required for __eq__ method.
    def _exact_match(self, other):
        if isinstance(other, self.__class__):
            # for split-antecedent we check all the members are the same
            if self.is_split_antecedent or other.is_split_antecedent:
                return self.split_antecedent_sets == other.split_antecedent_sets
            else:
                if len(self._words) != len(other._words):
                    return False
                words_zip = zip(self._words, other._words)
                return all(self_w == other_w for self_w, other_w in words_zip)
        return NotImplemented

    def _corefud_partial_match_score(self, other):
        """Partial matching method similar to the approach proposed by the CorefUD scorer.
        It allows the response to be part of the key and in the same time the response 
        must include all the words in MIN (head).
        If the above condition is satisfied, a non-zero similarity corresponding to the recall
        (`num_of_common_words/total_words_in_key`; precision is 100% by definition) is returned.
        Otherwise 0 is returned.
        Current default partial matching.
        """
        # MIN only annotated in the key
        if self._minset and self._minset.issubset(other._wordsset) and \
            other._wordsset.issubset(self._wordsset):
            return len(self._wordsset & other._wordsset) / len(self._wordsset)
        elif other._minset and other._minset.issubset(self._wordsset) and \
            self._wordsset.issubset(other._wordsset):
            return len(self._wordsset & other._wordsset) / len(other._wordsset)
        return 0

    def _craft_partial_match_score(self, other):
        """CRAFT partial matching used in the CRAFT 2019 CR task.
        It uses the first key span as the MIN and any response that overlaps with the MIN
        (start>=MIN[0] and end<=MIN[1]) receives a non-zero similarity score.
        Otherwise 0 is returned.
        Only supported in the UA format, yet.
        """
        return NotImplemented

    def _partial_match_score(self, other, method='default'):
        if isinstance(other, self.__class__):
            if self.__eq__(other):
                return 1
            if method.lower() == 'default':
                return self._corefud_partial_match_score(other)
            elif method.lower() == 'craft':
                return self._craft_partial_match_score(other)
        return NotImplemented

    def _zero_dependent_match_score(self,other):
        return NotImplemented

    def intersection(self, other):
        if isinstance(other, self.__class__):
            if self._words[0] > other._words[-1] or \
                other._words[0] > self._words[-1]:
                return []
            return self._wordsset.intersection(other._wordsset)
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            if self._words[0] == other._words[0]:
                if self._words[-1] == other._words[-1]:
                    return len(self._words) < len(other._words)
                else:
                    return self._words[-1] < other._words[-1]
            else:
                return self._words[0] < other._words[0]
        return NotImplemented

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __hash__(self):
        if self.is_split_antecedent:
            return hash(frozenset(self.split_antecedent_sets))
        return hash(frozenset((self._words)))

    def __str__(self):
        if self.is_split_antecedent:
            return "({:s})".format(",".join([str(cl[0]) for cl in self.split_antecedent_sets]))
        return "({:s})".format(
            ",".join([str(w) + "*" if self._minset and w in self._minset else str(w) for w in self._words]))

    def __repr__(self):
        return str(self)
