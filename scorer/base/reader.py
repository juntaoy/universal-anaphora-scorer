import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

class Reader:
    class DataAlignError(BaseException):
        def __init__(self, key_node, sys_node, misalign_source="Words"):
            self.key_node = key_node
            self.sys_node = sys_node
            self.misalign_source = misalign_source

        def __str__(self):
            return "{:s} in key and sys are not aligned: key={:s}, sys={:s}".format(
                self.misalign_source,
                str(self.key_node),
                str(self.sys_node))

    class CorefFormatError(BaseException):
        def __init__(self, message):
            self.message = message

        def __str__(self):
            return self.message

    def __init__(self,**kwargs):
        self._doc_coref_infos = {}
        self._doc_non_referring_infos={}
        self._doc_bridging_infos={}
        self._doc_discourse_deixis_infos = {}

        self.keep_singletons = kwargs.get("keep_singletons",False)
        self.keep_split_antecedents = kwargs.get("keep_split_antecedents",False)
        self.keep_bridging = kwargs.get("keep_bridging",False)
        self.keep_non_referring = kwargs.get("keep_non_referring",False)
        self.evaluate_discourse_deixis = kwargs.get("evaluate_discourse_deixis",False)
        self.partial_match = kwargs.get("partial_match",False)
        self.partial_match_method = kwargs.get("partial_match_method",'default')
        self.allow_boundary_crossing = kwargs.get("allow_boundary_crossing",False)
        self.np_only = kwargs.get('np_only',False)
        self.remove_nested_mentions = kwargs.get('remove_nested_mentions',False)

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


    def get_mention_cluster_alignment(self,clusters, other_mention_set):
        mention_cluster_ids = {}
        mention_non_aligned = []
        for cluster_id, cluster in enumerate(clusters):
            for m in cluster:
                if m in mention_cluster_ids:
                    logging.warning(
                        "Mention span {:s} has been already indexed with cluster_id = {:d}. New cluster_id = {:d}".format(
                            str(m), mention_cluster_ids[m], cluster_id))
                mention_cluster_ids[m] = cluster_id
                if not m.is_split_antecedent and m not in other_mention_set:
                    mention_non_aligned.append(m)
        return mention_cluster_ids, mention_non_aligned


    def get_mention_assignments(self,key_clusters, sys_clusters):
        key_mention_set = set([m for cl in key_clusters for m in cl])
        sys_mention_set = set([m for cl in sys_clusters for m in cl])
        s_num = len(key_mention_set & sys_mention_set)
        partial_match_dict = {}

        logging.debug('Total key mentions:', len(key_mention_set))
        logging.debug('Total response mentions:', len(sys_mention_set))
        logging.debug('Strictly correct indentified mentions:', s_num)

        #we no longer add partial matching to sys_mention_key_cluster or key_mention_sys_cluster the partial matching
        #is not handled by partial_match_dict alone.
        sys_mention_key_cluster, key_non_aligned = self.get_mention_cluster_alignment(key_clusters,sys_mention_set)
        key_mention_sys_cluster, sys_non_aligned = self.get_mention_cluster_alignment(sys_clusters,key_mention_set)

        use_CRAFT = self.partial_match_method == 'craft'
        p_num = 0
        if self.partial_match and len(key_non_aligned) > 0 and len(sys_non_aligned) > 0:  # partial matching
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
                                        partial_match_dict[sm] = km
                                        partial_match_dict[km] = sm
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
                        partial_match_dict[sys_mention] = key_mention
                        partial_match_dict[key_mention] = sys_mention
        logging.debug('Partially correct identified mentions:', p_num)
        logging.debug('No identified:', len(key_mention_set) - s_num - p_num)
        logging.debug('Invented:', len(sys_mention_set) - s_num - p_num)
        return sys_mention_key_cluster, key_mention_sys_cluster, partial_match_dict
