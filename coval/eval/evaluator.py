"""Some parts are borrowed from
https://github.com/clarkkev/deep-coref/blob/master/evaluation.py
"""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from coval.ua import markable


def f1(p_num, p_den, r_num, r_den, beta=1):
  p = 0 if p_den == 0 else p_num / float(p_den)
  r = 0 if r_den == 0 else r_num / float(r_den)
  return (0 if p + r == 0
      else (1 + beta * beta) * p * r / (beta * beta * p + r))

def evaluate_bridgings(doc_bridging_infos):
  tp_ar, fp_ar, fn_ar = 0,0,0 #anaphora recognation
  tp_fbm, fp_fbm, fn_fbm = 0,0,0 #full bridging at mention level
  tp_fbe, fp_fbe, fn_fbe = 0,0,0 #full bridging at entity level
  for doc_id in doc_bridging_infos:
    key_bridging_pairs, sys_bridging_pairs, mention_to_gold = doc_bridging_infos[doc_id]
    for k_ana in key_bridging_pairs:
      if k_ana in sys_bridging_pairs:
        tp_ar+=1
        k_ant = key_bridging_pairs[k_ana]
        s_ant = sys_bridging_pairs[k_ana]

        if k_ant == s_ant:
          tp_fbe+=1
          tp_fbm+=1
        else:
          fn_fbm+=1
          if s_ant in mention_to_gold and mention_to_gold[k_ant] == mention_to_gold[s_ant]:
            tp_fbe+=1
          else:
            fn_fbe+=1
      else:
        fn_ar+=1
        fn_fbe+=1
        fn_fbm+=1

    for s_ana in sys_bridging_pairs:
      if s_ana not in key_bridging_pairs:
        fp_ar+=1
        fp_fbe+=1
        fp_fbm+=1
      else:
        s_ant = sys_bridging_pairs[s_ana]
        k_ant = key_bridging_pairs[s_ana]
        if s_ant != k_ant:
          fp_fbm+=1
          if s_ant not in mention_to_gold or mention_to_gold[s_ant] != mention_to_gold[k_ant]:
            fp_fbe+=1
  recall_ar = tp_ar / float(tp_ar + fn_ar) if (tp_ar + fn_ar) > 0 else 0
  precision_ar = tp_ar / float(tp_ar + fp_ar) if (tp_ar + fp_ar) > 0 else 0
  f1_ar = (2 * recall_ar * precision_ar / (recall_ar + precision_ar)
        if (recall_ar + precision_ar) > 0 else 0)

  recall_fbm = tp_fbm / float(tp_fbm + fn_fbm) if (tp_fbm + fn_fbm) > 0 else 0
  precision_fbm = tp_fbm / float(tp_fbm + fp_fbm) if (tp_fbm + fp_fbm) > 0 else 0
  f1_fbm = (2 * recall_fbm * precision_fbm / (recall_fbm + precision_fbm)
            if (recall_fbm + precision_fbm) > 0 else 0)

  recall_fbe = tp_fbe / float(tp_fbe + fn_fbe) if (tp_fbe + fn_fbe) > 0 else 0
  precision_fbe = tp_fbe / float(tp_fbe + fp_fbe) if (tp_fbe + fp_fbe) > 0 else 0
  f1_fbe = (2 * recall_fbe * precision_fbe / (recall_fbe + precision_fbe)
        if (recall_fbe + precision_fbe) > 0 else 0)
  return (recall_ar, precision_ar, f1_ar), (recall_fbm, precision_fbm, f1_fbm), (recall_fbe, precision_fbe, f1_fbe)

def evaluate_non_referrings(doc_non_referring_infos):
  tp, _tn, fp, fn = 0, 0, 0, 0

  for doc_id in doc_non_referring_infos:
    key_non_referrings, sys_non_referrings = doc_non_referring_infos[
        doc_id]
    for m in key_non_referrings:
      if m in sys_non_referrings:
        tp += 1
      else:
        fn += 1
    for m in sys_non_referrings:
      if m not in key_non_referrings:
        fp += 1

  recall = tp / float(tp + fn) if (tp + fn) > 0 else 0
  precision = tp / float(tp + fp) if (tp + fp) > 0 else 0
  f1 = (2 * recall * precision / (recall + precision)
      if (recall + precision) > 0 else 0)

  return recall, precision, f1


class Evaluator:
  def __init__(self, metric, beta=1, keep_aggregated_values=False,lea_split_antecedent_importance=1):
    self.p_num = 0
    self.p_den = 0
    self.r_num = 0
    self.r_den = 0
    self.metric = metric
    self.beta = beta
    self.keep_aggregated_values = keep_aggregated_values
    self.lea_split_antecedent_importance = lea_split_antecedent_importance
    self.split_antecedent_counter = [0,0,0,0] #pn, pd, rn, rd

    if keep_aggregated_values:
      self.aggregated_p_num = []
      self.aggregated_p_den = []
      self.aggregated_r_num = []
      self.aggregated_r_den = []
  
  def align_split_antecedents(self, key_clusters, sys_clusters):
    key_split_antecedents = [m for cl in key_clusters for m in cl if is_split_antecedent(m)]
    sys_split_antecedents = [m for cl in sys_clusters for m in cl if is_split_antecedent(m)]

    if len(key_split_antecedents) == 0 or len(sys_split_antecedents) == 0:
      return {}, {}, {}

    key_clusters = [list(s_ant.split_antecedent_members) for s_ant in key_split_antecedents]
    sys_clusters = [list(s_ant.split_antecedent_members) for s_ant in sys_split_antecedents]
    sys_mention_key_clusters = [{m:cid for cid, cl in enumerate(clusters) for m in cl} for clusters in key_clusters]
    key_mention_sys_clusters = [{m:cid for cid, cl in enumerate(clusters) for m in cl} for clusters in sys_clusters]

    f_scores = np.zeros((len(key_split_antecedents), len(sys_split_antecedents)))
    recalls = np.zeros((len(key_split_antecedents), len(sys_split_antecedents)))
    precisions = np.zeros((len(key_split_antecedents), len(sys_split_antecedents)))
    raw_numbers = np.zeros((len(key_split_antecedents), len(sys_split_antecedents), 4)) #
    for i in range(len(key_split_antecedents)):
      key_cluster = key_clusters[i]
      sys_mention_key_cluster = sys_mention_key_clusters[i]
      for j in range(len(sys_split_antecedents)):
        sys_cluster = sys_clusters[j]
        key_mention_sys_cluster = key_mention_sys_clusters[j]
        pn, pd, rn, rd = self.__update__(key_cluster, sys_cluster,key_mention_sys_cluster,sys_mention_key_cluster,is_split_alignment=True)
        raw_numbers[i,j,:] = [pn, pd, rn, rd]
        precisions[i,j] = 0 if pn == 0 else pn / float(pd)
        recalls[i,j] = 0 if rn == 0 else rn / float(rd)
        f_scores[i,j] = f1(pn, pd, rn, rd)
    row_ind, col_ind = linear_sum_assignment(-f_scores)

    #pn,pd,rn,rd
    self.split_antecedent_counter[0] += raw_numbers[row_ind, col_ind, np.zeros_like(col_ind)].sum()
    self.split_antecedent_counter[1] += raw_numbers[0, :, 1].sum()
    self.split_antecedent_counter[2] += raw_numbers[row_ind, col_ind, np.ones_like(col_ind) * 2].sum()
    self.split_antecedent_counter[3] += raw_numbers[:, 0, 3].sum()

    key_split_antecedent_sys_r = {key_split_antecedents[r]: (sys_split_antecedents[c], float(recalls[r, c]))
                                  for r, c in zip(row_ind, col_ind) if recalls[r, c] > 0}
    sys_split_antecedent_key_p = {sys_split_antecedents[c]: (key_split_antecedents[r], float(precisions[r, c]))
                                  for r, c in zip(row_ind, col_ind) if precisions[r, c] > 0}
    key_split_antecedent_sys_f = {key_split_antecedents[r]: (sys_split_antecedents[c], float(f_scores[r, c]))
                                  for r, c in zip(row_ind, col_ind) if f_scores[r, c] > 0}

    return key_split_antecedent_sys_r, sys_split_antecedent_key_p, key_split_antecedent_sys_f

  def __update__(self, key_clusters, sys_clusters,
                 key_mention_sys_cluster, sys_mention_key_cluster,
                 key_split_antecedent_sys_r={},sys_split_antecedent_key_p={},
                 key_split_antecedent_sys_f={},is_split_alignment=False):
    if self.metric == ceafe or self.metric == ceafm:
      pn, pd, rn, rd = self.metric(sys_clusters, key_clusters,key_split_antecedent_sys_f)
    elif self.metric == blancc or self.metric == blancn:
      pn, pd, rn, rd = self.metric(sys_clusters, key_clusters, key_mention_sys_cluster, key_split_antecedent_sys_f)
    elif self.metric == lea:
      pn, pd = self.metric(sys_clusters, key_clusters,
          sys_mention_key_cluster,sys_split_antecedent_key_p,
                           self.lea_split_antecedent_importance)
      rn, rd = self.metric(key_clusters, sys_clusters,
          key_mention_sys_cluster,key_split_antecedent_sys_r,
                           self.lea_split_antecedent_importance)
    elif self.metric == muc:
      pn, pd = self.metric(sys_clusters, key_clusters, sys_mention_key_cluster, sys_split_antecedent_key_p,is_split_alignment)
      rn, rd = self.metric(key_clusters, sys_clusters, key_mention_sys_cluster, key_split_antecedent_sys_r,is_split_alignment)
    else:
      pn, pd = self.metric(sys_clusters, sys_mention_key_cluster, sys_split_antecedent_key_p)
      rn, rd = self.metric(key_clusters, key_mention_sys_cluster, key_split_antecedent_sys_r)

    # if pn != pd or rn != rd:
    #   print(pn,pd,rn,rd)
    #   for i, cl1 in enumerate(key_clusters):
    #     for cl2 in key_clusters[i+1:]:
    #       for m1 in cl1:
    #         for m2 in cl2:
    #           if m1 == m2:
    #             print(str(m1))

    return pn, pd, rn, rd

  def update(self, coref_info):
    (key_clusters, sys_clusters, key_mention_sys_cluster,
        sys_mention_key_cluster) = coref_info

    key_split_antecedent_sys_r, sys_split_antecedent_key_p, key_split_antecedent_sys_f \
      = self.align_split_antecedents(key_clusters, sys_clusters)

    pn, pd, rn, rd = self.__update__(key_clusters,sys_clusters,
                                     key_mention_sys_cluster,
                                     sys_mention_key_cluster,
                                     key_split_antecedent_sys_r,
                                     sys_split_antecedent_key_p,
                                     key_split_antecedent_sys_f)
    self.p_num += pn
    self.p_den += pd
    self.r_num += rn
    self.r_den += rd

    if self.keep_aggregated_values:
      self.aggregated_p_num.append(pn)
      self.aggregated_p_den.append(pd)
      self.aggregated_r_num.append(rn)
      self.aggregated_r_den.append(rd)

  def get_f1(self):
    return f1(self.p_num,
        self.p_den,
        self.r_num,
        self.r_den,
        beta=self.beta)

  def get_split_antecedent_prf(self):
    pn, pd, rn, rd = self.split_antecedent_counter
    p = 0 if pn == 0 else pn/float(pd)
    r = 0 if rn == 0 else rn/float(rd)
    return p, r, f1(pn,pd,rn,rd,beta=self.beta)

  def get_recall(self):
    return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

  def get_precision(self):
    return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

  def get_prf(self):
    return self.get_precision(), self.get_recall(), self.get_f1()

  def get_counts(self):
    return self.p_num, self.p_den, self.r_num, self.r_den

  def get_aggregated_values(self):
    return (self.aggregated_p_num, self.aggregated_p_den,
        self.aggregated_r_num, self.aggregated_r_den)


def evaluate_documents(doc_coref_infos, metric, beta=1, lea_split_antecedent_importance=1, only_split_antecedent = False):
  if isinstance(metric, list):
    #for blanc
    evaluators = [Evaluator(sub_metric,beta=beta,lea_split_antecedent_importance=lea_split_antecedent_importance) for sub_metric in metric]
    for doc_id in doc_coref_infos:
      for evaluator in evaluators:
        evaluator.update(doc_coref_infos[doc_id])
    p, r, f, cnt = 0,0,0,0
    for evaluator in evaluators:
      pn,pd,rn,rd = evaluator.get_counts()
      # print(pn,pd,rn,rd)
      if pd == rd == 0:
        continue
      sp, sr, sf = evaluator.get_split_antecedent_prf() if only_split_antecedent else evaluator.get_prf()
      p+=sp
      r+=sr
      f+=sf
      cnt+=1
    if cnt == 0:
      return 0,0,0
    else:
      return (r/cnt, p/cnt, f/cnt)
  else:
    evaluator = Evaluator(metric, beta=beta,lea_split_antecedent_importance=lea_split_antecedent_importance)
    for doc_id in doc_coref_infos:
      # print(doc_id)
      evaluator.update(doc_coref_infos[doc_id])
    if only_split_antecedent:
      p, r, f = evaluator.get_split_antecedent_prf()
      return r,p,f
    else:
      return (evaluator.get_recall(), evaluator.get_precision(),
        evaluator.get_f1())


def get_document_evaluations(doc_coref_infos, metric, beta=1):
  evaluator = Evaluator(metric, beta=beta, keep_aggregated_values=True)
  for doc_id in doc_coref_infos:
    evaluator.update(doc_coref_infos[doc_id])
  return evaluator.get_aggregated_values()


def mentions(clusters, mention_to_gold):
  setofmentions = set(mention for cluster in clusters for mention in cluster)
  correct = setofmentions & set(mention_to_gold.keys())
  return len(correct), len(setofmentions)


def b_cubed(clusters, mention_to_gold, split_antecedent_to_gold={}):
  num, den = 0, 0

  for c in clusters:
    gold_counts = defaultdict(float)
    correct = 0
    for m in c:
      if is_split_antecedent(m):
        if m in split_antecedent_to_gold:
          gold_split_antecedent, matching_score = split_antecedent_to_gold[m]
          gold_counts[mention_to_gold[gold_split_antecedent]] += matching_score
      elif m in mention_to_gold:
        gold_counts[mention_to_gold[m]] += 1
    for c2 in gold_counts:
      correct += gold_counts[c2] * gold_counts[c2]

    num += correct / float(len(c))
    den += len(c)

  return num, den

def is_split_antecedent(m):
  return isinstance(m, markable.Markable) and m.is_split_antecedent

def muc(clusters, out_clusters, mention_to_gold, split_antecedent_to_gold={}, count_singletons = False):
  tp, p = 0, 0
  for c in clusters:
    if len(c) == 1 and count_singletons:
      p+=1
      if c[0] in mention_to_gold and len(out_clusters[mention_to_gold[c[0]]]) == 1:
        tp+=1
    else:
      p += len(c) - 1
      tp += len(c)
      linked = set()
      split_antecedent = None
      for m in c:
        if is_split_antecedent(m):
          split_antecedent = m
        elif m in mention_to_gold:
          linked.add(mention_to_gold[m])
        else:
          tp -= 1
      if split_antecedent:
        if split_antecedent in split_antecedent_to_gold:
          gold_split_antecedent, matching_score = split_antecedent_to_gold[split_antecedent]
          gold_split_antecedent_cluster = mention_to_gold[gold_split_antecedent]
          if gold_split_antecedent_cluster in linked:
            tp -= 1 - matching_score
          else:
            tp -=1
        else:
          tp -=1
      tp -= len(linked)
  return tp, p


def phi4(c1, c2, split_antecedent_to_sys):
  return 2 * phi3(c1,c2,split_antecedent_to_sys) / float(len(c1) + len(c2))

def phi3(c1, c2, split_antecedent_to_sys):
  overlap = 0
  for m in c1:
    if is_split_antecedent(m):
      if m in split_antecedent_to_sys:
        gold_split_antecedent, matching_score = split_antecedent_to_sys[m]
        if gold_split_antecedent in c2:
          overlap += matching_score
    elif m in c2:
      overlap += 1
  return overlap

def ceafe(clusters, gold_clusters,key_split_antecedent_sys_f={}):
  clusters = [c for c in clusters]
  scores = np.zeros((len(gold_clusters), len(clusters)))
  for i in range(len(gold_clusters)):
    for j in range(len(clusters)):
      scores[i, j] = phi4(gold_clusters[i], clusters[j],key_split_antecedent_sys_f)
  row_ind, col_ind = linear_sum_assignment(-scores)
  # print(scores,row_ind,col_ind)
  similarity = scores[row_ind, col_ind].sum()
  return similarity, len(clusters), similarity, len(gold_clusters)

def ceafm(clusters, gold_clusters, key_split_antecedent_sys_f={}):
  clusters = [c for c in clusters]
  scores = np.zeros((len(gold_clusters), len(clusters)))
  for i in range(len(gold_clusters)):
    for j in range(len(clusters)):
      scores[i, j] = phi3(gold_clusters[i], clusters[j], key_split_antecedent_sys_f)
  row_ind, col_ind = linear_sum_assignment(-scores)
  similarity = scores[row_ind, col_ind].sum()

  #corrected by juntao for ceafm the denominator is the number of mentions
  #return similarity, len(clusters), similarity, len(gold_clusters)
  return similarity, sum([len(cl) for cl in clusters]), similarity, sum([len(cl) for cl in gold_clusters])


def lea(input_clusters, output_clusters, mention_to_gold, split_antecedent_to_gold={}, split_antecedent_importance=1):
  num, den = 0, 0

  for c in input_clusters:
    has_split_antecedent = False
    if len(c) == 1:
      all_links = 1
      if c[0] in mention_to_gold and len(
          output_clusters[mention_to_gold[c[0]]]) == 1:
        common_links = 1
      else:
        common_links = 0
    else:
      common_links = 0
      all_links = len(c) * (len(c) - 1) / 2.0
      for i, m in enumerate(c):
        link_score = 1
        if is_split_antecedent(m) and m in split_antecedent_to_gold:
          m, link_score = split_antecedent_to_gold[m]
          has_split_antecedent = True
        if m in mention_to_gold:
          for m2 in c[i + 1:]:
            if is_split_antecedent(m2) and m2 in split_antecedent_to_gold:
              m2, link_score2 = split_antecedent_to_gold[m2]
              has_split_antecedent = True
              link_score *=link_score2
            if m2 in mention_to_gold and mention_to_gold[
                m] == mention_to_gold[m2]:
              common_links += link_score
            # else:
            #  print('!! ', m2, '--', m2.get_span(), ' ',
            #       m2.min_spans, ' ', mention_to_gold[m], ' ',
            #       mention_to_gold[m2], ' ' ,
            #       [str(s) for s in output_clusters[
            #         mention_to_gold[m]]], ' -- ',
            #       [str(s) for s in output_clusters[
            #         mention_to_gold[m2]]])

    cluster_importance = (split_antecedent_importance if has_split_antecedent else 1)
    num += cluster_importance * len(c) * common_links / float(all_links)
    den += cluster_importance * len(c)
  return num, den


def blancc(sys_clusters, key_clusters, mention_to_sys, split_antecedent_to_sys_f={}):
  num, pd, rd = 0, 0, 0
  for c in key_clusters:
    common_links = 0
    for i, m in enumerate(c):
      link_score = 1
      if is_split_antecedent(m) and m in split_antecedent_to_sys_f:
        m, link_score = split_antecedent_to_sys_f[m]
      if m in mention_to_sys:
        for m2 in c[i + 1:]:
          if m2 in mention_to_sys and mention_to_sys[m] == mention_to_sys[m2]:
            common_links += link_score

    num += common_links
  rd = sum([len(c) * (len(c) - 1) / 2 for c in key_clusters])
  pd = sum([len(c) * (len(c) - 1) / 2 for c in sys_clusters])
  return num, pd, num, rd


def blancn(sys_clusters, key_clusters, mention_to_sys, split_antecedent_to_sys_f={}):
  num, pd, rd = 0, 0, 0
  for cid, c in enumerate(key_clusters):
    common_links = 0
    for i, m in enumerate(c):
      link_score = 1
      if is_split_antecedent(m) and m in split_antecedent_to_sys_f:
        m, link_score = split_antecedent_to_sys_f[m]
      if m in mention_to_sys:
        for c2 in key_clusters[cid + 1:]:
          for m2 in c2:
            link_score2 = 1
            if is_split_antecedent(m2) and m2 in split_antecedent_to_sys_f:
              m2, link_score2 = split_antecedent_to_sys_f[m2]
            if m2 in mention_to_sys and mention_to_sys[m] != mention_to_sys[m2]:
              common_links += link_score * link_score2

    num += common_links
  num_key_mentions = sum([len(c) for c in key_clusters])
  num_sys_mentions = sum([len(c) for c in sys_clusters])
  rd = num_key_mentions * (num_key_mentions - 1) / 2 - sum([len(c) * (len(c) - 1) / 2 for c in key_clusters])
  pd = num_sys_mentions * (num_sys_mentions - 1) / 2 - sum([len(c) * (len(c) - 1) / 2 for c in sys_clusters])
  return num, pd, num, rd