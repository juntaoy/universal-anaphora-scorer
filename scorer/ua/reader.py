import logging
from scorer.base.reader import Reader
from scorer.ua.mention import UAMention
from collections import deque, defaultdict

__author__ = 'ns-moosavi; juntaoy'


class UAReader(Reader):
    @property
    def doc_non_referring_infos(self):
        return self._doc_non_referring_infos

    @property
    def doc_bridging_infos(self):
        return self._doc_bridging_infos

    @property
    def doc_discourse_deixis_infos(self):
        return self._doc_discourse_deixis_infos

    def get_doc_markables(self, doc_name, doc_lines, word_column=1, markable_column=10, bridging_column=11):
        use_CRAFT_MIN = self.partial_match_method == 'craft'

        markables_cluster = {}
        markables_start = defaultdict(list)
        markables_end = defaultdict(list)
        markables_MIN = {}
        markables_coref_tag = {}
        markables_split = {}  # set_id: [markable_id_1, markable_id_2 ...]
        markables_is_zero={}
        bridging_antecedents = {}
        all_words = []
        stack = []


        word_index = 0
        # zero_index is used for distinguish multiple zeros in the same location
        # zero is index with decimals in string e.g. '5.0', '5.1'
        zero_index = 0
        for line in doc_lines:
            columns = line.split()
            is_zero = '.' in columns[0]

            if columns[markable_column] != '_':
                markable_annotations = columns[markable_column].split("(")
                if markable_annotations[0]:
                    # the close bracket
                    if is_zero:
                        raise self.CorefFormatError(f'Zeros should not be used as start/end of the standard mentions. {line}')
                    if self.allow_boundary_crossing:
                        for markable_id in markable_annotations[0].split(")"):
                            if len(markable_id) > 0:
                                markables_end[markable_id].append(word_index)
                    else:
                        for _ in range(len(markable_annotations[0])):
                            markable_id = stack.pop()
                            markables_end[markable_id].append(word_index)

                for markable_annotation in markable_annotations[1:]:
                    if markable_annotation.endswith(')'):
                        single_word = True
                        markable_annotation = markable_annotation[:-1]
                    else:
                        if is_zero:
                            raise self.CorefFormatError(
                                f'Zeros should not be used as start/end of the standard mentions. {line}')
                        single_word = False
                    markable_info = {p[:p.find('=')]: p[p.find('=') + 1:] for p in markable_annotation.split('|')}
                    markable_id = markable_info['MarkableID']
                    cluster_id = markable_info['EntityID']
                    markables_cluster[markable_id] = cluster_id
                    if markable_id not in markables_is_zero:
                        markables_is_zero[markable_id] = is_zero

                    if is_zero:
                        markables_start[markable_id].append('{}.{}'.format(word_index,zero_index))
                        markables_end[markable_id].append('{}.{}'.format(word_index,zero_index))
                    else:
                        markables_start[markable_id].append(word_index)
                        if single_word:
                            markables_end[markable_id].append(word_index)
                        elif not self.allow_boundary_crossing:
                            stack.append(markable_id)

                    if markable_id not in markables_MIN:
                        markables_MIN[markable_id] = None
                        if self.partial_match and 'Min' in markable_info:
                            MIN_Span = markable_info['Min'].split(',')
                            if len(MIN_Span) == 2:
                                MIN_start = int(MIN_Span[0]) - 1
                                MIN_end = int(MIN_Span[1]) - 1
                            else:
                                MIN_start = int(MIN_Span[0]) - 1
                                MIN_end = MIN_start
                            markables_MIN[markable_id] = [MIN_start, MIN_end]

                    markables_coref_tag[markable_id] = 'referring'
                    if cluster_id.endswith('-Pseudo'):
                        markables_coref_tag[markable_id] = 'non_referring'

                    if 'ElementOf' in markable_info:
                        element_of = markable_info['ElementOf'].split(
                            ',')  # for markable participate in multiple plural using , split the element_of, e.g. ElementOf=1,2
                        for ele_of in element_of:
                            if ele_of not in markables_split:
                                markables_split[ele_of] = []
                            markables_split[ele_of].append(markable_id)
            if self.keep_bridging and columns[bridging_column] != '_':
                bridging_annotations = columns[bridging_column].split("(")
                for bridging_annotation in bridging_annotations[1:]:
                    if bridging_annotation.endswith(')'):
                        bridging_annotation = bridging_annotation[:-1]
                    bridging_info = {p[:p.find('=')]: p[p.find('=') + 1:] for p in bridging_annotation.split('|')}
                    bridging_antecedents[bridging_info['MarkableID']] = bridging_info['MentionAnchor']

            if is_zero:
                zero_index+=1
            else:
                all_words.append(columns[word_column])
                zero_index=0
                word_index+=1

        clusters = {}
        id2markable = {}
        for markable_id in markables_cluster:
            m = UAMention(
                markables_start[markable_id],
                markables_end[markable_id],
                [markables_start[markable_id][0], markables_end[markable_id][0]] if use_CRAFT_MIN else markables_MIN[
                    markable_id],
                markables_coref_tag[markable_id],
                is_zero = markables_is_zero[markable_id]
            )
            id2markable[markable_id] = m
            if markables_cluster[markable_id] not in clusters:
                clusters[markables_cluster[markable_id]] = (
                    [], markables_coref_tag[markable_id], doc_name,
                    [markables_cluster[mid] for mid in markables_split.get(markables_cluster[markable_id], [])])
            clusters[markables_cluster[markable_id]][0].append(m)

        bridging_pairs = {}
        for anaphora, antecedent in bridging_antecedents.items():
            if not anaphora in id2markable or not antecedent in id2markable:
                logging.warning(
                    'Skip bridging pair ({}, {}) as markable_id does not exist in identity column!'.format(antecedent,
                                                                                                           anaphora))
                continue
            bridging_pairs[id2markable[anaphora]] = id2markable[antecedent]
        return clusters, bridging_pairs

    def process_clusters(self, clusters):
        removed_non_referring = 0
        removed_singletons = 0
        removed_zeros = 0
        processed_clusters = []
        processed_non_referrings = []

        for cluster_id, (cluster, ref_tag, doc_name, split_cid_list) in clusters.items():
            # recusively find the split singular cluster
            if split_cid_list and self.keep_split_antecedents:
                # if using split-antecedent, we shouldn't remove singletons as they might be used by split-antecedents
                assert self.keep_singletons
                split_clusters = set()
                queue = deque()
                queue.append(cluster_id)
                while queue:
                    curr = queue.popleft()
                    curr_cl, curr_ref_tag, doc_name, curr_cid_list = clusters[curr]
                    if curr_cid_list:
                        for c in curr_cid_list:
                            queue.append(c)
                    else:
                        split_clusters.add(tuple(curr_cl))
                split_m = UAMention(
                    [],
                    [], None,
                    'referring',
                    is_split_antecedent=True,
                    split_antecedent_sets=split_clusters)

                cluster.append(split_m)  # add the split_antecedents

            if ref_tag == 'non_referring':
                if self.keep_non_referring:
                    processed_non_referrings.append(cluster[0])
                else:
                    removed_non_referring += 1
                continue

            if not self.keep_singletons and len(cluster) == 1:
                removed_singletons += 1
                continue

            if not self.keep_zeros:
                o_size = len(cluster)
                cluster = [m for m in cluster if not m.is_zero]
                removed_zeros += o_size - len(cluster)

            processed_clusters.append(cluster)

        if self.keep_split_antecedents:
            # step 2 merge equivalent split-antecedents clusters
            merged_clusters = []
            for cl in processed_clusters:
                existing = None
                for m in cl:
                    if m.is_split_antecedent:
                        # only do this for split-antecedents
                        for c2 in merged_clusters:
                            if m in c2:
                                existing = c2
                                break
                if existing:
                    logging.warning('merge cluster ['+ ','.join([str(m) for m in cl])+ '] and ['+','.join([str(m) for m in existing])+']')
                    existing.update(cl)
                else:
                    merged_clusters.append(set(cl))
            merged_clusters = [list(cl) for cl in merged_clusters]
        else:
            merged_clusters = processed_clusters

        return (merged_clusters, processed_non_referrings,
                removed_non_referring, removed_singletons, removed_zeros)

    def get_coref_infos(self, key_file, sys_file, unit_test=False):
        key_docs = self.get_all_docs(key_file)
        sys_docs = self.get_all_docs(sys_file)

        self.check_data_alignment(key_docs, sys_docs, unit_test=unit_test)

        for doc in key_docs:
            markable_column = 12 if self.evaluate_discourse_deixis else 10
            key_clusters, key_bridging_pairs = self.get_doc_markables(doc, key_docs[doc],
                                                                      markable_column=markable_column)
            sys_clusters, sys_bridging_pairs = self.get_doc_markables(doc, sys_docs[doc],
                                                                      markable_column=markable_column)

            (key_clusters, key_non_referrings, key_removed_non_referring,
             key_removed_singletons,key_removed_zeros) = self.process_clusters(key_clusters)
            (sys_clusters, sys_non_referrings, sys_removed_non_referring,
             sys_removed_singletons,sys_removed_zeros) = self.process_clusters(sys_clusters)

            logging.debug(doc)

            sys_mention_key_cluster, key_mention_sys_cluster, partial_match_dict = self.get_mention_assignments(
                key_clusters, sys_clusters)

            if self.evaluate_discourse_deixis:
                self._doc_discourse_deixis_infos[doc] = (key_clusters, sys_clusters,
                                         key_mention_sys_cluster, sys_mention_key_cluster, partial_match_dict)
            else:
                self._doc_coref_infos[doc] = (key_clusters, sys_clusters,
                                         key_mention_sys_cluster, sys_mention_key_cluster, partial_match_dict)
            self._doc_non_referring_infos[doc] = (key_non_referrings, sys_non_referrings)
            self._doc_bridging_infos[doc] = (key_bridging_pairs, sys_bridging_pairs, sys_mention_key_cluster)

            if not self.keep_non_referring:
                logging.debug('%s and %s non-referring markables are removed from the '
                              'evaluations of the key and system files, respectively.'
                              % (key_removed_non_referring, sys_removed_non_referring))

            if not self.keep_singletons:
                logging.debug('%s and %s singletons are removed from the evaluations of '
                              'the key and system files, respectively.'
                              % (key_removed_singletons, sys_removed_singletons))

            if not self.keep_zeros:
                logging.debug('%s and %s zeros are removed from the evaluations of '
                              'the key and system files, respectively.'
                              % (key_removed_zeros, sys_removed_zeros))

    def get_all_docs(self, path):
        all_docs = {}
        doc_lines = []
        doc_name = None
        for line in open(path):
            line = line.strip()
            if line.startswith('# newdoc'):
                if doc_name and doc_lines:
                    all_docs[doc_name] = doc_lines
                    doc_lines = []
                doc_name = line[len('# newdoc id = '):]
            elif line.startswith('#') or len(line) == 0:
                continue
            else:
                doc_lines.append(line)
        if doc_name and doc_lines:
            all_docs[doc_name] = doc_lines
        return all_docs

    def get_doc_tokens_without_zeros(self, doc,word_column=1):
        tokens = []
        for line in doc:
            columns = line.split()
            if '.' not in columns[0]:
                tokens.append(columns[word_column])
        return tokens

    def check_data_alignment(self, key_docs, sys_docs, word_column=1, unit_test=False):
        if len(key_docs.keys()) != len(sys_docs.keys()) or \
            len(key_docs.keys() - sys_docs.keys()) > 0 or \
            len(sys_docs.keys() - key_docs.keys()) > 0:
            raise self.DataAlignError(key_docs.keys() - sys_docs.keys(), sys_docs.keys() - key_docs.keys(), "Documents", "doc missing in sys", "doc inserting in sys")

        for doc in key_docs.keys():
            key_tokens = self.get_doc_tokens_without_zeros(key_docs[doc], word_column)
            sys_tokens = self.get_doc_tokens_without_zeros(sys_docs[doc], word_column)
            if len(key_tokens) != len(sys_tokens):
                raise self.DataAlignError(len(key_tokens), len(sys_tokens), "Number of tokens (excluding zeros)")
            # for unit_test we do not check the actual tokens, as they may not be the same
            if not unit_test:
                for i, (kt, st) in enumerate(zip(key_tokens, sys_tokens)):
                    if kt != st:
                        raise self.DataAlignError(kt, st, f"Word {i+1}")
