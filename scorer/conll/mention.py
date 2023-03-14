import logging

from scorer.base.mention import Mention

class CoNLLMention(Mention):
    def __init__(self, sent_num, start, end):
        super().__init__()
        for w in range(start,end+1):
            self._words.append((sent_num,w))
        self._words.sort()
        self._wordsset = set(self._words)


        #class specific property
        self._gold_parse = None


    @property
    def sent_num(self):
        return self._words[0][0]

    @property
    def gold_parse(self):
        return self._gold_parse

    @property
    def gold_parse_is_set(self):
        return self._gold_parse is not None

    @gold_parse.setter
    def gold_parse(self, tree):
        self._gold_parse = tree

    def are_nested(self, other):
        if isinstance(other, self.__class__):
            if self.__eq__(other):
                return -1
            if True:
                #self is nested in other
                if self.sent_num == other.sent_num and \
                   self.start >= other.start and self.end <= other.end:
                    return 0
                #other is nested in self
                elif self.sent_num == other.sent_num and \
                   other.start >= self.start and other.end <= self.end:
                    return 1
                else:
                    return -1
       
        return NotImplemented


    '''
    This function is for specific cases in which the nodes 
    in the top two level of the mention parse tree do not contain a valid tag.
    E.g., (TOP (S (NP (NP one)(PP of (NP my friends)))))
    '''
    def get_min_span_no_valid_tag(self, root):
        if not root:
            return
        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        accepted_tags = None
    
        while queue:
            node, depth = queue.pop(0)

            if not accepted_tags:
                if node.tag[0:2] in ['NP', 'NM']:
                    accepted_tags=['NP', 'NM', 'QP', 'NX']
                elif node.tag[0:2]=='VP':
                    accepted_tags=['VP']

            if node.isTerminal and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.tag, node.pos):
                    self._minset.add((self.sent_num, node.index))
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)
                    
            elif (not self._minset or depth < terminal_shortest_depth)and node.children and \
                 (depth== 0 or not accepted_tags or node.tag[0:2] in accepted_tags): 
                for child in node.children:
                    if not child.isTerminal or (accepted_tags and node.tag[0:2] in accepted_tags):
                        queue.append((child, depth+1))


    """
    Exluding terminals like comma and paranthesis
    """
    def is_a_valid_terminal_node(self, tag, pos):
        if len(tag.split()) == 1:
            if (any(c.isalpha() for c in tag) or \
                any(c.isdigit() for c in tag) or tag == '%') \
                  and (tag != '-LRB-' and tag != '-RRB-') \
                  and pos[0] != 'CC' and pos[0] != 'DT' and pos[0] != 'IN':# not in conjunctions:
                return True
            return False
        else: # for exceptions like ", and"
            for i, tt in enumerate(tag.split()):
                if self.is_a_valid_terminal_node(tt, [pos[i]]):
                    return True
            return False
   

    def get_valid_node_min_span(self, root, valid_tags):
        if not root:
            return

        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        while queue:
            node, depth = queue.pop(0)

            if node.isTerminal and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.tag, node.pos):
                    self._minset.add((self.sent_num, node.index))
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)

            elif (not self._minset or depth < terminal_shortest_depth )and node.children and \
                 (depth== 0 or not valid_tags or node.tag[0:2] in valid_tags):
                for child in node.children:
                    if not child.isTerminal or (valid_tags and node.tag[0:2] in valid_tags):
                        queue.append((child, depth+1))


    def get_top_level_phrases(self, root, valid_tags):
        top_level_valid_phrases = []

        if root and root.isTerminal and self.is_a_valid_terminal_node(root.tag, root.pos):
            self._minset.add((self.sent_num, root.index))

        elif root and root.children:
            for node in root.children:
                if node:
                    if node.isTerminal and self.is_a_valid_terminal_node(node.tag, node.pos):
                        self._minset.add((self.sent_num, node.index))
            if not self._minset:
                for node in root.children:
                    if node.children and node.tag[0:2] in valid_tags:
                        top_level_valid_phrases.append(node)

        return top_level_valid_phrases

    def get_valid_tags(self, root):
        valid_tags = None
        NP_tags = ['NP', 'NM', 'QP', 'NX']
        VP_tags = ['VP']

        if root.tag[0:2]=='VP':
            valid_tags = VP_tags
        elif root.tag[0:2] in ['NP', 'NM']:
            valid_tags = NP_tags
        else:
            if root.children: ## If none of the first level nodes are either NP or VP, examines their children for valid mention tags
                all_tags = []
                for node in root.children:
                    all_tags.append(node.tag[0:2])
                if 'NP' in all_tags or 'NM' in all_tags:
                    valid_tags = NP_tags
                elif 'VP' in all_tags:
                    valid_tags = VP_tags
                else:
                    valid_tags = NP_tags

        return valid_tags


    def extract_min_span(self):

        if not self.gold_parse_is_set:
            logging.error('The parse tree should be set before extracting minimum spans')
            return NotImplemented

        root = self.gold_parse

        if not root:
            return

        valid_tags = self.get_valid_tags(root)


        top_level_valid_phrases = self.get_top_level_phrases(root, valid_tags)
        
        if self._minset:
            return
        '''
        In structures like conjunctions the minimum span is determined independently
        for each of the top-level NPs
        '''
        if top_level_valid_phrases:
            for node in top_level_valid_phrases:
                 self.get_valid_node_min_span(node, valid_tags)

        else:
            self.get_min_span_no_valid_tag(root)


        """
        If there was no valid minimum span due to parsing errors return the whole span
        """
        if len(self._minset)==0:
            self._minset = self._wordsset


    
class TreeNode:
    def __init__(self, tag, pos, index, isTerminal):
        self.tag = tag
        self.pos = pos
        self.index = index
        self.isTerminal = isTerminal
        self.children = []
        
    def __str__(self, level=0):
        ret = "\t"*level+(self.tag)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def get_terminals(self, terminals):
        if self.isTerminal:
            terminals.append(self.tag)
        else:
            for child in self.children:
                child.get_terminals(terminals)

    def refined_get_children(self):    
        children = []
        for child in self.children:
            if not child.isTerminal and child.children and len(child.children)==1 and child.children[0].isTerminal:
                children.append(child.children[0])
            else:
                children.append(child)
        return children

            
