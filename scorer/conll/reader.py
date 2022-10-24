import logging
from scorer.conll import mention as mention
from scorer.base.reader import Reader


class CoNLLReader(Reader):
    def get_doc_mentions(self, doc_lines, word_column=3):
        clusters = {}
        open_mentions = {}
        singletons_num = 0

        for sent_num, sent_line in enumerate(doc_lines):
            sent_words = []
            for word_index, line in enumerate(sent_line):

                sent_words.append(line.split()[word_column]
                                  if len(line.split()) > word_column + 1 else '')

                single_token_coref, open_corefs, end_corefs = (
                    self.extract_coref_annotation(line))

                if single_token_coref:
                    m = mention.CoNLLMention(sent_num, word_index, word_index)
                    for c in single_token_coref:
                        if c not in clusters:
                            clusters[c] = []
                        clusters[c].append(m)

                for c in open_corefs:
                    if c in open_mentions:
                        logging.debug('Nested coreferring mentions.\n' + str(line))
                        open_mentions[c].append([sent_num, word_index])
                    else:
                        open_mentions[c] = [[sent_num, word_index]]

                for c in end_corefs:
                    if c not in clusters:
                        clusters[c] = []
                    if c not in open_mentions:
                        logging.warning('Problem in the coreference annotation:\n'+ line)
                    else:
                        if open_mentions[c][0][0] != sent_num:
                            logging.warning('A mention span should be in a single sentence:')
                            logging.warning(line)

                        m = mention.CoNLLMention(
                            sent_num, open_mentions[c][-1][1],
                            word_index)
                        clusters[c].append(m)
                        if len(open_mentions[c]) == 1:
                            open_mentions.pop(c)
                        else:
                            open_mentions[c].pop()

        if not self.keep_singletons:
            singletons = []
            for c in clusters:
                if len(clusters[c]) == 1:
                    singletons.append(c)
            singletons_num += len(singletons)
            for c in sorted(singletons, reverse=True):
                clusters.pop(c)

        return [clusters[c] for c in clusters], singletons_num

    def mask_unseen_mentions(self, clusters, seen_mentions):
        unseens = {}

        for i, cluster in enumerate(clusters):
            for m in cluster:
                if m not in seen_mentions:
                    if i not in unseens:
                        unseens[i] = set()
                    unseens[i].add(m)

        remove_clusters = set()
        for i in unseens:
            clusters[i] = [m for m in clusters[i] if m not in unseens[i]]

            if (len(clusters[i]) == 0
                or (len(clusters[i]) == 1 and not self.keep_singletons)):
                remove_clusters.add(i)

        return [c for i, c in enumerate(clusters) if i not in remove_clusters]

    def extract_coref_annotation(self, line):
        single_token_coref = []
        open_corefs = []
        ending_corefs = []
        last_num = []
        coref_opened = False

        coref_column = line.split()[-1]

        for i, c in enumerate(coref_column):
            if c.isdigit():
                last_num.append(c)
            elif c == '(':
                last_num = []
                coref_opened = True
            elif c == ')':
                if coref_opened:
                    # Coreference annotations that are marked without specifying
                    # the chain number will be skipped
                    if len(last_num) > 0:
                        single_token_coref.append(int(''.join(last_num)))
                    coref_opened = False
                    last_num = []
                else:
                    if len(last_num) > 0:
                        ending_corefs.append(int(''.join(last_num)))
                        last_num = []
            elif c == '|':
                if coref_opened:
                    open_corefs.append(int(''.join(last_num)))
                    coref_opened = False
                    last_num = []
                elif len(last_num) > 0:
                    raise self.CorefFormatError("Incorrect coreference annotation: " + coref_column)

            if i == len(coref_column) - 1:
                if coref_opened and len(last_num) > 0:
                    open_corefs.append(int(''.join(last_num)))

        if len(single_token_coref) > 1:
            logging.warning('A single mention is assigned to more than one cluster: %s'
                            % single_token_coref)

        return single_token_coref, open_corefs, ending_corefs

    def extract_annotated_parse(self, mention_lines, start_index,
                                parse_column=5, word_column=3, POS_column=4):
        """Extracting gold parse annotation according to the CoNLL format."""
        open_nodes = []
        tag_started = False
        tag_name = []
        terminal_nodes = []
        pos_tags = []
        root = None
        roots = []

        for i, line in enumerate(mention_lines):
            parse = line.split()[parse_column]
            for j, c in enumerate(parse):
                if c == '(':
                    if tag_started:
                        node = mention.TreeNode(''.join(tag_name), pos_tags,
                                                start_index + i, False)
                        if open_nodes:
                            if open_nodes[-1].children:
                                open_nodes[-1].children.append(node)
                            else:
                                open_nodes[-1].children = [node]

                        open_nodes.append(node)
                        tag_name = []
                    if terminal_nodes:
                        # skipping words like commas, quotations and parantheses
                        if any(c.isalpha() for c in terminal_nodes) or \
                            any(c.isdigit() for c in terminal_nodes):
                            node = mention.TreeNode(' '.join(terminal_nodes),
                                                    pos_tags, start_index + i, True)
                            if open_nodes:
                                if open_nodes[-1].children:
                                    open_nodes[-1].children.append(node)
                                else:
                                    open_nodes[-1].children = [node]
                            else:
                                open_nodes.append(node)
                        terminal_nodes = []
                        pos_tags = []

                    tag_started = True

                elif c == '*':
                    terminal_nodes.append(line.split()[word_column])
                    pos_tags.append(line.split()[POS_column])
                    node = mention.TreeNode(''.join(tag_name), None,
                                            start_index + i, False)

                    if tag_started:
                        if open_nodes:
                            if open_nodes[-1].children:
                                open_nodes[-1].children.append(node)
                            else:
                                open_nodes[-1].children = [node]

                        open_nodes.append(node)
                        tag_name = []
                        tag_started = False

                    elif tag_name:
                        roots.append(node)

                elif c == ')':
                    if terminal_nodes:
                        node = mention.TreeNode(' '.join(terminal_nodes),
                                                pos_tags, start_index + i, True)
                        if open_nodes:
                            if open_nodes[-1].children:
                                open_nodes[-1].children.append(node)
                            else:
                                open_nodes[-1].children = [node]
                        else:
                            open_nodes.append(node)

                        terminal_nodes = []
                        pos_tags = []

                    if open_nodes:
                        root = open_nodes.pop()
                        if not open_nodes:
                            roots.append(root)

                    tag_started = False

                elif c.isalpha():
                    tag_name.append(c)

                if (i == len(mention_lines) - 1 and
                    j == len(parse) - 1 and terminal_nodes):
                    node = mention.TreeNode(' '.join(terminal_nodes),
                                            pos_tags, start_index + i, True)
                    if open_nodes:
                        if open_nodes[-1].children:
                            open_nodes[-1].children.append(node)
                        else:
                            open_nodes[-1].children = [node]
                    else:
                        open_nodes.append(node)

                    terminal_nodes = []
                    pos_tags = []

        # If there is parsing errors in which starting phrasea are not ended at
        # the end of detected mention boundaries
        while open_nodes:
            root = open_nodes.pop()
            if not open_nodes:
                roots.append(root)

        if len(roots) > 1:
            new_root = mention.TreeNode('NP', None, start_index, False)
            for node in roots:
                new_root.children.append(node)
            return new_root

        return root

    def set_annotated_parse_trees(self, clusters, key_doc_lines, partial_vp_chain_pruning=True):
        pruned_cluster_indices = set()
        pruned_clusters = {}

        for i, c in enumerate(clusters):
            pruned_cluster = list(c)
            for m in c:
                try:
                    tree = self.extract_annotated_parse(
                        key_doc_lines[m.sent_num][m.start:m.end + 1], m.start)
                except IndexError as err:
                    logging.error(str(err) + '\n {},{}'.format(len(key_doc_lines), m.sent_num))

                m.gold_parse = tree

                ##If the conll file does not have words
                if not m.words[0]:
                    terminals = []
                    m.gold_parse.get_terminals(terminals)
                    m.words = []
                    for t in terminals:
                        for w in t.split():
                            m.words.append(w)

                if self.partial_match:
                    m.extract_min_span()
                if tree and tree.tag == 'VP' and self.np_only:
                    pruned_cluster.remove(m)
                    pruned_cluster_indices.add(i)
            pruned_clusters[i] = pruned_cluster

        if self.np_only and pruned_cluster_indices:
            for i in sorted(pruned_cluster_indices, reverse=True):
                if len(pruned_clusters[i]) > 1 and partial_vp_chain_pruning:
                    logging.debug('VP partial pruning: '+
                              ','.join([str(m) for m in clusters[i]]) + '->' +
                              ','.join([str(m) for m in pruned_clusters[i]]))
                else:
                    logging.debug('VP full pruning, cluster size: '+ str(len(clusters[i]))+
                              ' cluster: ' + ','.join([str(m) for m in clusters[i]]))
                    pruned_clusters.pop(i)

        return [pruned_clusters[k] for k in pruned_clusters]

    def get_doc_lines(self, file_name):
        doc_lines = {}
        doc_name = None

        with open(file_name) as f:
            new_sentence = True
            for line in f:
                if line.startswith("#begin document"):
                    doc_name = line[len("#begin document "):]
                elif line.startswith("#end document"):
                    doc_name = None

                elif doc_name:
                    if doc_name not in doc_lines:
                        doc_lines[doc_name] = []
                    if (not line.strip()
                        and not new_sentence) or not doc_lines[doc_name]:
                        doc_lines[doc_name].append([])

                    if line.strip():
                        new_sentence = False
                        doc_lines[doc_name][-1].append(line)
                    else:
                        new_sentence = True

        return doc_lines

    def remove_nested_coref_mentions(self, clusters):
        to_be_removed_mentions = {}
        to_be_removed_clusters = []
        all_removed_mentions = 0
        all_removed_clusters = 0

        for c_index, c in enumerate(clusters):
            to_be_removed_mentions[c_index] = []

            for i, m1 in enumerate(c):
                for m2 in c[i + 1:]:
                    nested = m1.are_nested(m2)
                    # m1 is nested in m2
                    if nested == 0:
                        to_be_removed_mentions[c_index].append(m1)
                        logging.debug(','.join([m1, m2]))
                        logging.debug('=========================')
                    # m2 is nested in m1
                    elif nested == 1:
                        to_be_removed_mentions[c_index].append(m2)
                        logging.debug(m2)

        for c_index in to_be_removed_mentions:
            all_removed_mentions += len(to_be_removed_mentions[c_index])

            if len(clusters[c_index]) != 1 and len(clusters[c_index]) - len(
                to_be_removed_mentions[c_index]) == 1:
                all_removed_clusters += 1

                logging.debug(clusters[c_index][0])

                if not self.keep_singletons:
                    to_be_removed_clusters.append(c_index)
            else:
                clusters[c_index] = [
                    m for m in clusters[c_index]
                    if m not in to_be_removed_mentions[c_index]
                ]

        for c_index in sorted(to_be_removed_clusters, reverse=True):
            clusters.pop(c_index)

        return all_removed_mentions, all_removed_clusters

    def get_coref_infos(self, key_file, sys_file):

        key_doc_lines = self.get_doc_lines(key_file)
        sys_doc_lines = self.get_doc_lines(sys_file)

        key_nested_coref_num = 0
        sys_nested_coref_num = 0
        key_removed_nested_clusters = 0
        sys_removed_nested_clusters = 0
        key_singletons_num = 0
        sys_singletons_num = 0

        for doc in key_doc_lines:

            key_clusters, singletons_num = self.get_doc_mentions(key_doc_lines[doc])
            key_singletons_num += singletons_num

            if self.np_only or self.partial_match:
                key_clusters = self.set_annotated_parse_trees(key_clusters,
                                                              key_doc_lines[doc])

            sys_clusters, singletons_num = self.get_doc_mentions(sys_doc_lines[doc])
            sys_singletons_num += singletons_num

            if self.np_only or self.partial_match:
                sys_clusters = self.set_annotated_parse_trees(sys_clusters,
                                                              key_doc_lines[doc])

            if self.remove_nested_mentions:
                nested_mentions, removed_clusters = self.remove_nested_coref_mentions(
                    key_clusters)
                key_nested_coref_num += nested_mentions
                key_removed_nested_clusters += removed_clusters

                nested_mentions, removed_clusters = self.remove_nested_coref_mentions(sys_clusters)
                sys_nested_coref_num += nested_mentions
                sys_removed_nested_clusters += removed_clusters

            sys_mention_key_cluster, key_mention_sys_cluster, partial_match_dict = self.get_mention_assignments(
                key_clusters, sys_clusters)

            self._doc_coref_infos[doc] = (key_clusters, sys_clusters,
                                          key_mention_sys_cluster, sys_mention_key_cluster, partial_match_dict)

        if self.remove_nested_mentions:
            logging.warning('Number of removed nested coreferring mentions in the key '
                            'annotation: %s; and system annotation: %s' % (
                                key_nested_coref_num, sys_nested_coref_num))
            logging.warning('Number of resulting singleton clusters in the key '
                            'annotation: %s; and system annotation: %s' % (
                                key_removed_nested_clusters, sys_removed_nested_clusters))

        if not self.keep_singletons:
            logging.warning('%d and %d singletons are removed from the key and system '
                            'files, respectively' % (key_singletons_num, sys_singletons_num))
