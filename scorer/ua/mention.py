from scorer.base.mention import Mention
from scorer.base.reader import CorefFormatError

class UAMention(Mention):
    def __init__(self, start, end, MIN, is_referring, is_split_antecedent=False, split_antecedent_sets=set(), is_zero=False):
        super().__init__()

        if is_zero:
            if not len(start) == len(end) == 1:
                raise CorefFormatError(f'Zero mentions cannot be discontinuous: {list(zip(start, end))}')
            if start[0] != end[0]:
                raise CorefFormatError(f'Zero must consist of a single token: ({start[0]}, {end[0]})')
            self._words.append(start[0])
        else:
            for s, e in zip(start, end):
                for w in range(s, e + 1):
                    # include all words in [s,e] both inclusive
                    # any possible zeros in between are ignored
                    self._words.append(w)
            self._words.sort()
        self._wordsset = set(self._words)
        self._is_referring = is_referring
        self._is_split_antecedent = is_split_antecedent
        self._split_antecedent_sets = split_antecedent_sets
        self._is_zero = is_zero
        if MIN:
            self._minset = set(range(MIN[0], MIN[1] + 1))

        # class specific property for craft
        self._start_list = start
        self._end_list = end
        self._min = MIN

    # CRAFT (with craft tag) same as the CRAFT 2019 CR task that uses the first key span as the MIN and any
    #             response that is covered by the MIN (start>=MIN[0] and end <=MIN[1]) will receive a
    #             non-zero similarity score otherwise a zero will be returned.
    def _craft_partial_match_score(self, other):
        if self.is_zero or other.is_zero:
            return 0.0

        if self._minset:
            for s, e in zip(other._start_list, other._end_list):
                if s >= self._min[0] and e <= self._min[1]:
                    return len(self._wordsset & other._wordsset) / len(self._wordsset)
        elif other._minset:
            for s, e in zip(self._start_list, self._end_list):
                if s >= other._min[0] and e <= other._min[1]:
                    return len(self._wordsset & other._wordsset) / len(other._wordsset)
        return 0
