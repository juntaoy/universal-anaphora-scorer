from scorer.base.mention import Mention

class UAMention(Mention):
    def __init__(self, start, end, MIN, is_referring, is_split_antecedent=False, split_antecedent_sets=set()):
        super().__init__()

        for s, e in zip(start, end):
            for w in range(s, e + 1):
                self._words.append(w)  # [s,e] both inclusive
        self._words.sort()
        self._wordsset = set(self._words)
        self._is_referring = is_referring
        self._is_split_antecedent = is_split_antecedent
        self._split_antecedent_sets = split_antecedent_sets
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
        if self._minset:
            for s, e in zip(other._start_list, other._end_list):
                if s >= self._min[0] and e <= self._min[1]:
                    return len(self._wordsset & other._wordsset) / len(self._wordsset)
        elif other._minset:
            for s, e in zip(self._start_list, self._end_list):
                if s >= other._min[0] and e <= other._min[1]:
                    return len(self._wordsset & other._wordsset) / len(other._wordsset)
        return 0
