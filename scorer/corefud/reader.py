import logging
from udapi.core.document import Document
from udapi.block.read.conllu import Conllu
from collections import defaultdict, OrderedDict
from scorer.corefud.mention import CorefUDMention
from scorer.base.reader import Reader




class CorefUDReader(Reader):

    def load_conllu(self, file_path):
        doc = Document()
        conllu_reader = Conllu(files=[file_path])
        conllu_reader.apply_on_document(doc)
        return doc

    def check_data_alignment(self, data1, data2):
        data12_trees = zip(data1.trees, data2.trees)
        for tree1, tree2 in data12_trees:
            if tree1.newdoc != tree2.newdoc:
                raise self.DataAlignError(tree1, tree2, "Newdoc labels")
            if tree1.sent_id != tree2.sent_id:
                raise self.DataAlignError(tree1, tree2, "Sent IDs")
            data12_nodes = zip(tree1.descendants_and_empty, tree2.descendants_and_empty)
            for node1, node2 in data12_nodes:
                if node1.form != node2.form:
                    raise self.DataAlignError(node1, node2, "Words")

    def split_data_to_docs(self, data):
        word2docid = {}
        docord = 0
        docid = None
        doc_clusters = defaultdict(list)
        for tree in data.trees:
            if tree.newdoc:
                docord += 1
                docid = tree.newdoc if tree.newdoc is not True else docord
                doc_clusters[docid] = defaultdict(list)
            for node in tree.descendants_and_empty:
                word2docid[node] = docid

        for cluster in data.coref_entities:
            mention_doc = None
            for mention in cluster.mentions:
                words_docs = list(set([word2docid[w] for w in mention.words]))
                if len(words_docs) > 1:
                    mention_str = ", ".join([str(w) for w in mention.words])
                    raise self.CorefFormatError(
                        "Mention cannot cross a document boundary. The following does: " + mention_str)
                if mention_doc and mention_doc != words_docs[0]:
                    logging.warning(
                        f"Cluster {cluster.eid} spans two documents ({mention_doc}, {words_docs[0]}). It will be split.")
                mention_doc = words_docs[0]
                doc_clusters[mention_doc][cluster.eid].append(mention)
        return doc_clusters

    def transform_clusters_for_eval(self, clusters, nohead=False):
        # TODO: CorefMentions sets the first mention's word as its head if no head is specified
        # This is problematic for partial matching of key and sys mention.
        # For continuous mentions, a key mention is partially matched by the sys mention,
        # if the sys mention lies within the key mention and covers its head/min at the same time.
        # It requires the sys mention to have no head/min annotated.
        # All key mentions in CorefUD data should have their head annotated as they are automatically
        # added based on the dependency tree. This is, however, a result of running the MoveHead block.
        # Otherwise, the first mention node is declared to be its head.
        # Users applying UDAPI for their own data does not have to know it and thus the partial
        # matching is not going to work properly for them.
        transformed_clusters = []
        for cluster in clusters.values():
            transformed_cluster = [CorefUDMention(m.words, None if nohead else m.head) for m in cluster]
            # TODO: evaluator tests (TC-A-7.response) require to delete duplicate mention spans
            transformed_cluster = list(OrderedDict.fromkeys(transformed_cluster))
            transformed_clusters.append(transformed_cluster)
        return transformed_clusters

    def process_clusters(self, clusters):
        removed_singletons = 0
        processed_clusters = []
        for cluster in clusters:
            if not self.keep_singletons and len(cluster) == 1:
                removed_singletons += 1
                continue
            processed_clusters.append(cluster)
        return processed_clusters, removed_singletons

    def get_coref_infos(self, key_file, sys_file):
        # loading the documents
        key_data = self.load_conllu(key_file)
        sys_data = self.load_conllu(sys_file)

        # checking if key and sys data are aligned
        self.check_data_alignment(key_data, sys_data)

        # split data into documents and collect the clusters per document
        # also checking if relations do not cross document boundaries
        key_doc_clusters = self.split_data_to_docs(key_data)
        sys_doc_clusters = self.split_data_to_docs(sys_data)

        for docname in key_doc_clusters:
            assert docname in sys_doc_clusters

            key_clusters = self.transform_clusters_for_eval(key_doc_clusters[docname], not self.partial_match)
            sys_clusters = self.transform_clusters_for_eval(sys_doc_clusters[docname], True)

            key_clusters, key_removed_singletons = self.process_clusters(key_clusters)
            sys_clusters, sys_removed_singletons = self.process_clusters(sys_clusters)

            key_mention_to_cluster, sys_mention_to_cluster, partial_match_dict = self.get_mention_assignments(
                key_clusters, sys_clusters)

            # for an unknown reason, scorer.eval expects the tuple where
            # key_mention_to_cluster and sys_mention_to_cluster are
            # in the opposite order than key_cluster and sys_cluster
            self._doc_coref_infos[docname] = (key_clusters, sys_clusters,
                                              sys_mention_to_cluster, key_mention_to_cluster, partial_match_dict)
            if not self.keep_singletons:
                logging.debug(
                    "Singletons removed: key={:d}, sys={:d}".format(key_removed_singletons, sys_removed_singletons))
