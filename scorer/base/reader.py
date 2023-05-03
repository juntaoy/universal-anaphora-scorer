import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

class DataAlignError(BaseException):
    def __init__(self, key_node, sys_node, misalign_source="Words", key_name='key', sys_name='sys'):
        self.key_node = key_node
        self.sys_node = sys_node
        self.misalign_source = misalign_source
        self.key_name = key_name
        self.sys_name = sys_name

    def __str__(self):
        return f"{self.misalign_source} in key and sys are not aligned: \
                    {self.key_name}={str(self.key_node)}, {self.sys_name}={str(self.sys_node)}"

class CorefFormatError(BaseException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

class Reader:

    def __init__(self,**kwargs):
        self._doc_coref_infos = {}
        self._doc_non_referring_infos = {}
        self._doc_bridging_infos = {}
        self._doc_discourse_deixis_infos = {}

        self.keep_singletons = kwargs.get("keep_singletons", False)
        self.keep_split_antecedents = kwargs.get("keep_split_antecedents", False)
        self.keep_bridging = kwargs.get("keep_bridging", False)
        self.keep_non_referring = kwargs.get("keep_non_referring", False)
        self.evaluate_discourse_deixis = kwargs.get("evaluate_discourse_deixis", False)
        self.partial_match = kwargs.get("partial_match", False)
        self.partial_match_method = kwargs.get("partial_match_method", 'default')
        self.keep_zeros = kwargs.get("keep_zeros", False)
        self.zero_match_method = kwargs.get("zero_match_method", 'linear')
        self.allow_boundary_crossing = kwargs.get("allow_boundary_crossing", False)
        self.np_only = kwargs.get('np_only', False)
        self.remove_nested_mentions = kwargs.get('remove_nested_mentions', False)

    #the minimum requirement is to implement the coreference part
    @property
    def doc_coref_infos(self):
        return self._doc_coref_infos

    @property
    def doc_non_referring_infos(self):
        return NotImplemented

    @property
    def doc_bridging_infos(self):
        return NotImplemented

    @property
    def doc_discourse_deixis_infos(self):
        return NotImplemented

    def get_coref_infos(self, key_file, sys_file):
        return NotImplemented


    def get_mention_cluster_alignment(self, clusters, other_mention_set):
        mention_cluster_ids = {}
        mention_non_aligned = []
        for cluster_id, cluster in enumerate(clusters):
            for m in cluster:
                if m in mention_cluster_ids:
                    logging.warning(f"Mention span {str(m)} has been already indexed with cluster_id = {mention_cluster_ids[m]}. New cluster_id = {cluster_id}")
                mention_cluster_ids[m] = cluster_id
                if not m.is_split_antecedent and not m.is_zero and m not in other_mention_set:
                    mention_non_aligned.append(m)
        return mention_cluster_ids, mention_non_aligned


    def get_mention_assignments(self, key_clusters, sys_clusters):
        key_mention_set = set([m for cl in key_clusters for m in cl])
        sys_mention_set = set([m for cl in sys_clusters for m in cl])
        if self.keep_zeros and self.zero_match_method == 'dependent':
            s_num = len([ m for m in key_mention_set & sys_mention_set if not m.is_zero])
        else:
            s_num = len(key_mention_set & sys_mention_set)

        # the dict is shared between zeros for dependent alignment
        # method and the non-zeros mentions's partial matching
        mention_alignment_dict = {}

        if self.keep_zeros and self.zero_match_method == 'dependent':
            key_zeros = [m for m in key_mention_set if m.is_zero]
            sys_zeros = [m for m in sys_mention_set if m.is_zero]
            if len(key_zeros) > 0 and len(sys_zeros) > 0:
                key_zeros.sort()
                sys_zeros.sort()
                similarity = np.zeros((len(key_zeros), len(sys_zeros)))
                for i, km in enumerate(key_zeros):
                    for j, sm in enumerate(sys_zeros):
                        similarity[i, j] = km._zero_dependent_match_score(sm)
                # print(similarity)
                key_ind, sys_ind = linear_sum_assignment(-similarity)
                for k, s in zip(key_ind, sys_ind):
                    if similarity[k, s] > 0:
                        s_num += 1
                        key_mention, sys_mention = key_zeros[k], sys_zeros[s]
                        mention_alignment_dict[sys_mention] = key_mention
                        mention_alignment_dict[key_mention] = sys_mention

        logging.debug('Total key mentions:', len(key_mention_set))
        logging.debug('Total response mentions:', len(sys_mention_set))
        logging.debug('Strictly matched mentions:', s_num)

        # exact matching
        # partially matched mentions are not indexed by sys_mention_key_cluster and key_mention_sys_cluster
        # instead, mention_alignment_dict is used for partial matching
        sys_mention_key_cluster, key_non_aligned = self.get_mention_cluster_alignment(key_clusters, sys_mention_set)
        key_mention_sys_cluster, sys_non_aligned = self.get_mention_cluster_alignment(sys_clusters, key_mention_set)

        # partial matching
        use_CRAFT = self.partial_match_method == 'craft'
        p_num = 0
        if self.partial_match and len(key_non_aligned) > 0 and len(sys_non_aligned) > 0:
            # sort the mentions in order by start and end indices so that the KM algorithm can make
            # the alignment using same rule as corefUD:
            # 1. pick the mention that overlaps with m with proportionally smallest difference
            # 2. if still more than one n remain, pick the one that starts earlier in the document
            # 3. if still more than one n remain, pick the one that ends earlier in the document
            # 1 were done using similarity score based on proportional token overlapping,
            # 2 and 3 were done by sorting so that the mentions were sorted with the starts and ends.
            key_non_aligned.sort()
            sys_non_aligned.sort()
            if use_CRAFT:
                key_used = {km: False for km in key_non_aligned}
                key_non_aligned = set(key_non_aligned)
                for scl in sys_clusters:
                    for sm in scl:
                        if sm in sys_non_aligned:
                            for km in key_non_aligned:
                                # if not key_used[j] and km.similarity_scores(sm, method='craft') > 0:
                                if km._partial_match_score(sm, method='craft') > 0:
                                    if not key_used[km]:
                                        key_used[km] = True
                                        # print(str(km), str(sm))
                                        p_num += 1
                                        # sys_mention_key_cluster[sm] = sys_mention_key_cluster[km]
                                        # key_mention_sys_cluster[km] = key_mention_sys_cluster[sm]
                                        mention_alignment_dict[sm] = km
                                        mention_alignment_dict[km] = sm
                                    break
            else:
                similarity = np.zeros((len(key_non_aligned), len(sys_non_aligned)))
                for i, km in enumerate(key_non_aligned):
                    for j, sm in enumerate(sys_non_aligned):
                        similarity[i, j] = km._partial_match_score(sm, method=self.partial_match_method)
                # print(similarity)
                key_ind, sys_ind = linear_sum_assignment(-similarity)
                for k, s in zip(key_ind, sys_ind):
                    if similarity[k, s] > 0:
                        p_num += 1
                        key_mention, sys_mention = key_non_aligned[k], sys_non_aligned[s]
                        # print(str(key_mention),str(sys_mention))
                        # sys_mention_key_cluster[sys_mention] = sys_mention_key_cluster[key_mention]
                        # key_mention_sys_cluster[key_mention] = key_mention_sys_cluster[sys_mention]
                        mention_alignment_dict[sys_mention] = key_mention
                        mention_alignment_dict[key_mention] = sys_mention
        logging.debug('Partially matched mentions:', p_num)
        logging.debug('Unmatched key mentions:', len(key_mention_set) - s_num - p_num)
        logging.debug('Spurious response mentions:', len(sys_mention_set) - s_num - p_num)


        return sys_mention_key_cluster, key_mention_sys_cluster, mention_alignment_dict
