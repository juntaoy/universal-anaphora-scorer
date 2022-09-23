def get_dummy_split_antecedent():
  return Markable('', [], [], None,'referring','',True)

class Markable:
  def __init__(self, doc_name, start, end, MIN, is_referring, words,is_split_antecedent=False,split_antecedent_members=set()):
    self.doc_name = doc_name
    self.start = start #list
    self.end = end #list
    self.wordsets = set()
    for s,e in zip(self.start,self.end):
      for w in range(s,e+1):
        self.wordsets.add(w) #[s,e] both inclusive
    self.MIN = MIN #[Min_start, Min_end]
    self.MINsets = set(range(self.MIN[0],self.MIN[1]+1)) if self.MIN else None
    self.is_referring = is_referring
    self.words = words
    self.is_split_antecedent = is_split_antecedent
    self.split_antecedent_members = split_antecedent_members

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      # for split-antecedent we check all the members are the same
      if self.is_split_antecedent or other.is_split_antecedent:
        return self.split_antecedent_members == other.split_antecedent_members
      # # MIN is only set for the key markables
      # elif self.MIN:
      #   return (self.doc_name == other.doc_name
      #       and other.start >= self.start
      #       and other.start <= self.MIN[0]
      #       and other.end <= self.end
      #       and other.end >= self.MIN[1])
      # elif other.MIN:
      #   return (self.doc_name == other.doc_name
      #       and self.start >= other.start
      #       and self.start <= other.MIN[0]
      #       and self.end <= other.end
      #       and self.end >= other.MIN[1])
      else:
        return (self.doc_name == other.doc_name
            and self.start == other.start
            and self.end == other.end)
    return NotImplemented



  #Default (with MIN tag) similar to the CorefUD that allow the response to be part of the key, in the
  #             sametime the response must include all the words in MIN(head), if the above condition is
  #             satisfied then a non-zero similarity score based on the proportion of the common words
  #             (num_of_common_words/total_words_in_key) will be returned otherwise 0 will be returned.
  #CRAFT (with craft tag) same as the CRAFT 2019 CR task that use the first key span as the MIN and any
  #             response that overlapping with the MIN (start>=MIN[0] and end <=MIN[1]) will receive a
  #             non-zero similarity score otherwise a zero will be returned.
  def similarity_scores(self, other, method = 'default'):
    if isinstance(other, self.__class__):
      if self.__eq__(other):
        return 1.0
      if self.doc_name != other.doc_name:
        return 0.0

      if method.lower() == 'default':
        if self.MIN and self.MINsets.issubset(other.wordsets) and other.wordsets.issubset(self.wordsets): #MIN only annotated in the key
          return len(self.wordsets & other.wordsets)*1.0/len(self.wordsets)
        elif other.MIN and other.MINsets.issubset(self.wordsets) and self.wordsets.issubset(other.wordsets):
          return len(self.wordsets & other.wordsets)*1.0/len(other.wordsets)

        return 0.0
      elif method.lower() == 'craft':
        if self.MIN:
          for s,e in zip(other.start,other.end):
            if s >= self.MIN[0] and e <= self.MIN[1]:
              return len(self.wordsets & other.wordsets) * 1.0 / len(self.wordsets)
        elif other.MIN:
          for s,e in zip(self.start,self.end):
            if s>=other.MIN[0] and e <= other.MIN[1]:
              return len(self.wordsets & other.wordsets)*1.0/len(other.wordsets)

        return 0.0

    return NotImplemented

  def __lt__(self,other):
    if isinstance(other,self.__class__):
      if min(self.start) != min(other.start):
        return min(self.start) < min(other.start)
      else:
        return max(self.end) < max(other.end)

  def __le__(self, other):
    return self.__lt__(other) or self.__eq__(other)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    if self.is_split_antecedent:
      return hash(frozenset(self.split_antecedent_members))
    return hash(frozenset((self.start + self.end)))

  def __short_str__(self):
    return ('({},{})'.format(self.start,self.end))

  def __str__(self):
    if self.is_split_antecedent:
      return str([cl[0].__short_str__() for cl in self.split_antecedent_members])
    return self.__short_str__()

