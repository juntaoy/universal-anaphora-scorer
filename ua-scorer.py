import sys
from coval.ua import reader
from coval.eval import evaluator
from coval.eval.evaluator import evaluate_non_referrings

__author__ = 'ns-moosavi; juntaoy'


def main():
  metric_dict = {
      'lea': evaluator.lea, 'muc': evaluator.muc,
      'bcub': evaluator.b_cubed, 'ceafe': evaluator.ceafe,
      'ceafm':evaluator.ceafm, 'blanc':[evaluator.blancc,evaluator.blancn]}
  key_file = sys.argv[1]
  sys_file = sys.argv[2]

  if 'remove_singletons' in sys.argv or 'remove_singleton' in sys.argv:
    keep_singletons = False
  else:
    keep_singletons = True

  if 'remove_split_antecedent' in sys.argv or 'remove_split_antecedents' in sys.argv:
    keep_split_antecedent = False
  else:
    keep_split_antecedent = True

  if 'MIN' in sys.argv or 'min' in sys.argv or 'min_spans' in sys.argv:
    use_MIN = True
  else:
    use_MIN = False

  if 'keep_non_referring' in sys.argv or 'keep_non_referrings' in sys.argv:
    keep_non_referring = True
  else:
    keep_non_referring = False

  if 'keep_bridging' in sys.argv or 'keep_bridgings' in sys.argv:
    keep_bridging = True
  else:
    keep_bridging = False

  if 'only_split_antecedent' in sys.argv or 'only_split_antecedents' in sys.argv:
    only_split_antecedent = True
    keep_split_antecedent = True
    keep_singletons = True
    keep_bridging = False
    keep_non_referring=False
  else:
    only_split_antecedent = False

  if 'evaluate_discourse_deixis' in sys.argv:
    evaluate_discourse_deixis = True
    keep_split_antecedent = True
    keep_singletons = True
    only_split_antecedent = False
    keep_bridging = False
    keep_non_referring = False
  else:
    evaluate_discourse_deixis = False

  if 'all' in sys.argv:
    metrics = [(k, metric_dict[k]) for k in metric_dict]
  else:
    metrics = []
    for name in metric_dict:
      if name in sys.argv:
        
        metrics.append((name, metric_dict[name]))

  if len(metrics) == 0:
    metrics = [(name, metric_dict[name]) for name in metric_dict]

  msg = ""
  if evaluate_discourse_deixis:
    msg = 'only discourse deixis'
  elif only_split_antecedent:
    msg = 'only split-antecedents'
  else:
    msg = 'corferent markables'
    if keep_singletons:
      msg+= ', singletons'
    if keep_split_antecedent:
      msg+=', split-antecedents'
    if keep_non_referring:
      msg+=', non-referring mentions'
    if keep_bridging:
      msg+=', bridging relations'


  print('The scorer is evaluating ', msg,
      (" using the minimum span evaluation setting " if use_MIN else ""))

  evaluate(key_file, sys_file, metrics, keep_singletons,keep_split_antecedent,keep_bridging,
      keep_non_referring,only_split_antecedent,evaluate_discourse_deixis, use_MIN)


def evaluate(key_file, sys_file, metrics, keep_singletons, keep_split_antecedent, keep_bridging,
    keep_non_referring, only_split_antecedent,evaluate_discourse_deixis, use_MIN):

  doc_coref_infos, doc_non_referring_infos, doc_bridging_infos = reader.get_coref_infos(key_file, sys_file, keep_singletons,
      keep_split_antecedent, keep_bridging, keep_non_referring,evaluate_discourse_deixis,use_MIN)

  conll = 0
  conll_subparts_num = 0

  for name, metric in metrics:  
    recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,
        metric,
        beta=1,
        only_split_antecedent=only_split_antecedent)
    if name in ["muc", "bcub", "ceafe"]:
      conll += f1
      conll_subparts_num += 1

    print(name)
    print('Recall: %.2f' % (recall * 100),
        ' Precision: %.2f' % (precision * 100),
        ' F1: %.2f' % (f1 * 100))

  if conll_subparts_num == 3:
    conll = (conll / 3) * 100
    print('CoNLL score: %.2f' % conll)

  if keep_non_referring:
    recall, precision, f1 = evaluate_non_referrings(
        doc_non_referring_infos)
    print('============================================')
    print('Non-referring markable identification scores:')
    print('Recall: %.2f' % (recall * 100),
        ' Precision: %.2f' % (precision * 100),
        ' F1: %.2f' % (f1 * 100))
  if keep_bridging:
    score_ar, score_fbm, score_fbe = evaluator.evaluate_bridgings(doc_bridging_infos)
    recall_ar, precision_ar, f1_ar = score_ar
    recall_fbm, precision_fbm, f1_fbm = score_fbm
    recall_fbe, precision_fbe, f1_fbe = score_fbe

    print('============================================')
    print('Bridging anaphora recognition scores:')
    print('Recall: %.2f' % (recall_ar * 100),
          ' Precision: %.2f' % (precision_ar * 100),
          ' F1: %.2f' % (f1_ar * 100))
    print('Full bridging scores (Markable Level):')
    print('Recall: %.2f' % (recall_fbm * 100),
          ' Precision: %.2f' % (precision_fbm * 100),
          ' F1: %.2f' % (f1_fbm * 100))
    print('Full bridging scores (Entity Level):')
    print('Recall: %.2f' % (recall_fbe * 100),
          ' Precision: %.2f' % (precision_fbe * 100),
          ' F1: %.2f' % (f1_fbe * 100))

main()
