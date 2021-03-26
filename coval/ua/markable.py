class Markable:
  def __init__(self, doc_name, start, end, MIN, is_referring, words,is_split_antecedent=False,split_antecedent_members=set()):
    self.doc_name = doc_name
    self.start = start
    self.end = end
    self.MIN = MIN
    self.is_referring = is_referring
    self.words = words
    self.is_split_antecedent = is_split_antecedent
    self.split_antecedent_members = split_antecedent_members

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      # for split-antecedent we check all the members are the same
      if self.is_split_antecedent or other.is_split_antecedent:
        return self.split_antecedent_members == other.split_antecedent_members
      # MIN is only set for the key markables
      elif self.MIN:
        return (self.doc_name == other.doc_name
            and other.start >= self.start
            and other.start <= self.MIN[0]
            and other.end <= self.end
            and other.end >= self.MIN[1])
      elif other.MIN:
        return (self.doc_name == other.doc_name
            and self.start >= other.start
            and self.start <= other.MIN[0]
            and self.end <= other.end
            and self.end >= other.MIN[1])
      else:
        return (self.doc_name == other.doc_name
            and self.start == other.start
            and self.end == other.end)
    return NotImplemented

  def __neq__(self, other):
    if isinstance(other, self.__class__):
      return self.__eq__(other)
    return NotImplemented

  def __hash__(self):
    if self.is_split_antecedent:
      return hash(frozenset(self.split_antecedent_members))
    return hash(frozenset((self.start, self.end)))

  def __short_str__(self):
    return ('({},{})'.format(self.start,self.end))

  def __str__(self):
    if self.is_split_antecedent:
      return str([cl[0].__short_str__() for cl in self.split_antecedent_members])
    return self.__short_str__()
      # ('DOC: %s SPAN: (%d, %d) String: %r MIN: %s Referring tag: %s'
      #   % (
      #     self.doc_name, self.start, self.end, ' '.join(self.words),
      #     '(%d, %d)' % self.MIN if self.MIN else '',
      #     self.is_referring))
