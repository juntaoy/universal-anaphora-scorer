from os import walk
from os.path import isfile, join
from coval.ua import markable
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

__author__ = 'ns-moosavi; juntaoy'

def get_doc_markables(doc_name, doc_lines, extract_MIN, use_CRAFT_MIN, keep_bridging, word_column=1,
    markable_column=10, bridging_column=11, allow_boundary_crossing=False, print_debug=False):
  markables_cluster = {}
  markables_start = {}
  markables_end = {}
  markables_MIN = {}
  markables_coref_tag = {}
  markables_split = {} # set_id: [markable_id_1, markable_id_2 ...]
  bridging_antecedents = {}
  all_words = []
  stack = []
  for word_index, line in enumerate(doc_lines):
    columns = line.split()
    all_words.append(columns[word_column])

    if columns[markable_column] != '_':
      markable_annotations = columns[markable_column].split("(")
      if markable_annotations[0]:
        #the close bracket
        if allow_boundary_crossing:
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
          single_word = False
        markable_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in markable_annotation.split('|')}
        markable_id = markable_info['MarkableID']
        cluster_id = markable_info['EntityID']
        markables_cluster[markable_id] = cluster_id
        if markable_id not in markables_start:
          markables_start[markable_id] = []
          markables_end[markable_id] = []

        markables_start[markable_id].append(word_index)
        if single_word:
          markables_end[markable_id].append(word_index)
        elif not allow_boundary_crossing:
          stack.append(markable_id)

        if markable_id not in markables_MIN:
          markables_MIN[markable_id] = None
          if extract_MIN and 'Min' in markable_info:
            MIN_Span = markable_info['Min'].split(',')
            if len(MIN_Span) == 2:
              MIN_start = int(MIN_Span[0]) - 1
              MIN_end = int(MIN_Span[1]) - 1
            else:
              MIN_start = int(MIN_Span[0]) - 1
              MIN_end = MIN_start
            markables_MIN[markable_id] = [MIN_start,MIN_end]

        markables_coref_tag[markable_id] = 'referring'
        if cluster_id.endswith('-Pseudo'):
          markables_coref_tag[markable_id] = 'non_referring'

        if 'ElementOf' in markable_info:
          element_of = markable_info['ElementOf'].split(',') # for markable participate in multiple plural using , split the element_of, e.g. ElementOf=1,2
          for ele_of in element_of:
            if ele_of not in markables_split:
              markables_split[ele_of] = []
            markables_split[ele_of].append(markable_id)
    if keep_bridging and columns[bridging_column] != '_':
      bridging_annotations = columns[bridging_column].split("(")
      for bridging_annotation in bridging_annotations[1:]:
        if bridging_annotation.endswith(')'):
          bridging_annotation = bridging_annotation[:-1]
        bridging_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in bridging_annotation.split('|')}
        bridging_antecedents[bridging_info['MarkableID']] = bridging_info['MentionAnchor']




  clusters = {}
  id2markable = {}
  for markable_id in markables_cluster:
    m = markable.Markable(
        doc_name, markables_start[markable_id],
        markables_end[markable_id], [markables_start[markable_id][0],markables_end[markable_id][0]] if use_CRAFT_MIN else markables_MIN[markable_id],
        markables_coref_tag[markable_id],
        [all_words[s:e+1] for s,e in zip(markables_start[markable_id],markables_end[markable_id])])
    id2markable[markable_id] = m
    if markables_cluster[markable_id] not in clusters:
      clusters[markables_cluster[markable_id]] = (
          [], markables_coref_tag[markable_id],doc_name,[markables_cluster[mid] for mid in markables_split.get(markables_cluster[markable_id],[])])
    clusters[markables_cluster[markable_id]][0].append(m)

  bridging_pairs = {}
  for anaphora, antecedent in bridging_antecedents.items():
    if not anaphora in id2markable or not antecedent in id2markable:
      print('Skip bridging pair ({}, {}) as markable_id does not exist in identity column!'.format(antecedent,anaphora))
      continue
    bridging_pairs[id2markable[anaphora]] = id2markable[antecedent]

  #print([(str(ana),str(ant)) for ana,ant in bridging_pairs.items()])
  # for cid in clusters:
  #   cl = clusters[cid]
  #   print(cid,[str(m) for m in cl[0]],cl[1],cl[2],cl[3] )
  return clusters, bridging_pairs


def process_clusters(clusters, keep_singletons, keep_non_referring,keep_split_antecedent):
  removed_non_referring = 0
  removed_singletons = 0
  processed_clusters = []
  processed_non_referrings = []

  for cluster_id, (cluster, ref_tag, doc_name, split_cid_list) in clusters.items():
    #recusively find the split singular cluster
    if split_cid_list and keep_split_antecedent:
      # if using split-antecedent, we shouldn't remove singletons as they might be used by split-antecedents
      assert keep_singletons
      split_clusters = set()
      queue = deque()
      queue.append(cluster_id)
      while queue:
        curr = queue.popleft()
        curr_cl, curr_ref_tag, doc_name, curr_cid_list = clusters[curr]
        #non_referring shouldn't be used as split-antecedents
        # if curr_ref_tag != 'referring':
        #   print(curr_ref_tag, doc_name, curr_cid_list)
        if curr_cid_list:
          for c in curr_cid_list:
            queue.append(c)
        else:
          split_clusters.add(tuple(curr_cl))
      split_m = markable.Markable(
        doc_name, [],
        [], None,
        'referring',
        '',
        is_split_antecedent=True,
        split_antecedent_members=split_clusters)

      cluster.append(split_m) #add the split_antecedents

    if ref_tag == 'non_referring':
      if keep_non_referring:
        processed_non_referrings.append(cluster[0])
      else:
        removed_non_referring += 1
      continue

    if not keep_singletons and len(cluster) == 1:
      removed_singletons += 1
      continue

    processed_clusters.append(cluster)

  if keep_split_antecedent:
    #step 2 merge equivalent split-antecedents clusters
    merged_clusters = []
    for cl in processed_clusters:
      existing = None
      for m in cl:
        if m.is_split_antecedent:
        #only do this for split-antecedents
          for c2 in merged_clusters:
            if m in c2:
              existing = c2
              break
      if existing:
        # print('merge cluster ', [str(m) for m in cl], ' and ', [str(m) for m in existing])
        existing.update(cl)
      else:
        merged_clusters.append(set(cl))
    merged_clusters = [list(cl) for cl in merged_clusters]
  else:
    merged_clusters = processed_clusters

  return (merged_clusters, processed_non_referrings,
      removed_non_referring, removed_singletons)


def get_coref_infos(key_file,
    sys_file,
    keep_singletons,
    keep_split_antecedent,
    keep_bridging,
    keep_non_referring,
    evaluate_discourse_deixis,
    use_MIN,
    use_CRAFT,
    allow_boundary_crossing=False,
    print_debug=False):
  key_docs = get_all_docs(key_file)
  sys_docs = get_all_docs(sys_file)

  doc_coref_infos = {}
  doc_non_referrig_infos = {}
  doc_bridging_infos = {}

  for doc in key_docs:

    if doc not in sys_docs:
      print('The document ', doc,
          ' does not exist in the system output.')
      continue
    markable_column = 12 if evaluate_discourse_deixis else 10
    key_clusters, key_bridging_pairs = get_doc_markables(doc, key_docs[doc], use_MIN,use_CRAFT, keep_bridging,markable_column=markable_column,allow_boundary_crossing=allow_boundary_crossing)
    sys_clusters, sys_bridging_pairs = get_doc_markables(doc, sys_docs[doc], False,False, keep_bridging,markable_column=markable_column,allow_boundary_crossing=allow_boundary_crossing)

    (key_clusters, key_non_referrings, key_removed_non_referring,
        key_removed_singletons) = process_clusters(
        key_clusters, keep_singletons, keep_non_referring,keep_split_antecedent)
    (sys_clusters, sys_non_referrings, sys_removed_non_referring,
        sys_removed_singletons) = process_clusters(
        sys_clusters, keep_singletons, keep_non_referring,keep_split_antecedent)
    if print_debug:
      print(doc)
    sys_mention_key_cluster,key_mention_sys_cluster,partial_match_map = get_markable_assignments(key_clusters,sys_clusters,use_MIN,use_CRAFT,print_debug)

    doc_coref_infos[doc] = (key_clusters, sys_clusters,
          key_mention_sys_cluster, sys_mention_key_cluster,partial_match_map)
    doc_non_referrig_infos[doc] = (key_non_referrings, sys_non_referrings)
    doc_bridging_infos[doc] = (key_bridging_pairs, sys_bridging_pairs, sys_mention_key_cluster)

    if print_debug and not keep_non_referring:
      print('%s and %s non-referring markables are removed from the '
          'evaluations of the key and system files, respectively.'
          % (key_removed_non_referring, sys_removed_non_referring))

    if print_debug and not keep_singletons:
      print('%s and %s singletons are removed from the evaluations of '
          'the key and system files, respectively.'
          % (key_removed_singletons, sys_removed_singletons))

  return doc_coref_infos, doc_non_referrig_infos, doc_bridging_infos



def get_markable_assignments(key_clusters,sys_clusters,use_MIN,use_CRAFT,print_debug=False):
  key_mention_set = set([m for cl in key_clusters for m in cl])
  sys_mention_set = set([m for cl in sys_clusters for m in cl])
  s_num = len(key_mention_set&sys_mention_set)
  partial_match_map = {}
  if print_debug:
    print('Total key mentions:',len(key_mention_set))
    print('Total response mentions:',len(sys_mention_set))
    print('Strictly correct indentified mentions:',s_num)
  key_non_aligned = []
  sys_non_aligned = []

  sys_mention_key_cluster = {}
  for cluster_id, cluster in enumerate(key_clusters):
    for m in cluster:
      sys_mention_key_cluster[m] = cluster_id
      if not m.is_split_antecedent and m not in sys_mention_set:
        key_non_aligned.append(m)
  key_mention_sys_cluster = {}
  for cluster_id, cluster in enumerate(sys_clusters):
    for m in cluster:
      key_mention_sys_cluster[m] = cluster_id
      if not m.is_split_antecedent and m not in key_mention_set:
        sys_non_aligned.append(m)

  p_num = 0
  if use_MIN and len(key_non_aligned) > 0 and len(sys_non_aligned)>0: #partial matching
    # sort the mentions in order by start and end indices so that the KM algorithm can make
    # the alignment using same rule as corefUD:
    # 1. pick the mention that overlaps with m with proportionally smallest difference
    # 2. if still more than one n remain, pick the one that starts earlier in the document
    # 3. if still more than one n remain, pick the one that ends earlier in the document
    # 1 were done using similarity score based on proportional token overlapping,
    # 2 and 3 were done by sorting so that the mentions were sorted with the starts and ends.
    key_non_aligned.sort()
    sys_non_aligned.sort()
    # print([str(key) for key in key_non_aligned])
    # print([str(sys) for sys in sys_non_aligned])
    if use_CRAFT:
      key_used = {km:False for km in key_non_aligned}
      key_non_aligned = set(key_non_aligned)
      for scl in sys_clusters:
        for sm in scl:
          if sm in sys_non_aligned:
            for km in key_non_aligned:
              # if not key_used[j] and km.similarity_scores(sm, method='craft') > 0:
              if km.similarity_scores(sm, method='craft') > 0:
                if not key_used[km]:
                  key_used[km] = True
                  # print(str(km), str(sm))
                  p_num+=1
                  sys_mention_key_cluster[sm] = sys_mention_key_cluster[km]
                  key_mention_sys_cluster[km] = key_mention_sys_cluster[sm]
                  partial_match_map[sm] = km
                  partial_match_map[km] = sm
                break
    else:
      similarity = np.zeros((len(key_non_aligned),len(sys_non_aligned)))
      for i, km in enumerate(key_non_aligned):
        for j, sm in enumerate(sys_non_aligned):
          similarity[i,j] = km.similarity_scores(sm,method='craft' if use_CRAFT else 'default')
      # print(similarity)
      key_ind, sys_ind = linear_sum_assignment(-similarity)
      for k, s in zip(key_ind,sys_ind):
        if similarity[k,s] > 0:
          p_num+=1
          key_mention,sys_mention = key_non_aligned[k],sys_non_aligned[s]
          # print(str(key_mention),str(sys_mention))
          sys_mention_key_cluster[sys_mention] = sys_mention_key_cluster[key_mention]
          key_mention_sys_cluster[key_mention] = key_mention_sys_cluster[sys_mention]
          partial_match_map[sm] = km
          partial_match_map[km] = sm
  if print_debug:
    print('Partially correct identified mentions:',p_num)
    print('No identified:',len(key_mention_set)-s_num-p_num)
    print('Invented:',len(sys_mention_set)-s_num-p_num)
  return sys_mention_key_cluster, key_mention_sys_cluster,partial_match_map


def get_all_docs(path):
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





