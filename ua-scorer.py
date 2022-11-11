import sys, argparse, logging
from scorer.ua.reader import UAReader
from scorer.corefud.reader import CorefUDReader
from scorer.conll.reader import CoNLLReader
from scorer.eval import evaluator
from scorer.eval.evaluator import evaluate_non_referrings

__author__ = 'ns-moosavi; juntaoy; michnov'


class UnSuporttedFunctionError(BaseException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def compatibility_check(args):
    error_msg = ''
    format = args['format']
    format_specific_tags = {
        "ua": ['keep_split_antecedents', 'only_split_antecedent', 'evaluate_discourse_deixis',
               'allow_boundary_crossing'],
        "conll": ['np_only', 'remove_nested_mentions']
    }
    format_specific_metrics = {
        "ua": ['non-referring', 'bridging'],
        "corefud": ["zero"]
    }

    for target_format in format_specific_tags:
        specific_tags = format_specific_tags[target_format]
        if any([args[tag] for tag in specific_tags]) and format != target_format:
            error_msg += 'One or more options [{:s}] are only available for {:s} format.\n'.format(
                ','.join([tag for tag in specific_tags if args[tag]]),
                target_format
            )
    for target_format in format_specific_metrics:
        specific_metrics = format_specific_metrics[target_format]
        if any([tag in args['metrics'] for tag in specific_metrics]) and format != target_format:
            error_msg += 'One or more metrics [{:s}] are only available for {:s} format.\n'.format(
                ','.join([tag for tag in specific_metrics if tag in args['metrics']]),
                target_format
            )

    if args['partial_match'] and args['partial_match_method'] == 'craft' and format != 'ua':
        error_msg += 'The craft partial match method is only available for ua format.\n'

    if error_msg:
        raise UnSuporttedFunctionError(error_msg)


def autoreset_msg(key, value, parent):
    logging.warning(
        'Auto reset: {:s} must be {:b} when {:s} is used, reset to required value.'.format(key, value, parent))


def metric_autoremove_msg(key, parent):
    logging.warning(
        'Metric {:s} can not been used in conjunction with {:s}, removed from the evaluation.'.format(key, parent))


SHARED_TASK_SETTINGS = {
    "conll12": {
        "format": 'conll',
        "metrics": ['muc', 'bcub', 'ceafe']
    },
    "crac18": {
        "format": 'ua',
        "metrics": ['muc', 'bcub', 'ceafe', 'non-referring'],
        "keep_singletons": True
    },
    "craft19": {
        "format": 'ua',
        "metrics": ['muc', 'bcub', 'ceafe'],
        "keep_singletons": True,  # ??
        "keep_split_antecedents": False,
        "partial_match": True,  # ??
        "partial_match_method": 'craft'
    },
    "crac22": {
        "format": 'corefud',
        "metrics": ['muc', 'bcub', 'ceafe', 'mention', 'zero'],
        "keep_singletons": True,
        "partial_match": True,
        "partial_match_method": 'default'
    },
    "codicrac22ar": {
        "format": 'ua',
        "metrics": ['muc', 'bcub', 'ceafe'],
        "keep_singletons": True,
        "keep_split_antecedents": True
    },
    "codicrac22dd": {
        "format": 'ua',
        "metrics": ['muc', 'bcub', 'ceafe'],
        "keep_singletons": True,
        "keep_split_antecedents": True,
        "evaluate_discourse_deixis": True
    },
    "codicrac22br": {
        "format": 'ua',
        "metrics": ["bridging"],
        "keep_singletons": True,
    }
}


def main():
    argparser = argparse.ArgumentParser(description="Universal Anaphora scorer v2.0")
    argparser.add_argument('key_file', type=str, help='path to the key/reference file')
    argparser.add_argument('sys_file', type=str, help='path to the system/response file')
    argparser.add_argument('-f', '--format', choices=['ua', 'corefud', 'conll'], default='ua',
                           help='the input format for the scorer.')
    argparser.add_argument('-m', '--metrics',
                           choices=['all', 'conll', 'muc', 'bcub', 'ceafe', 'ceafm', 'blanc', 'lea', 'mention', 'zero',
                                    'non-referring', 'bridging'],
                           nargs='*', default='conll',
                           help='metrics to be used for evaluation, conll=avg[muc, bcub, ceafe]')
    argparser.add_argument('-s', '--keep-singletons', action='store_true', default=False,
                           help='evaluate also singletons; ignored otherwise')
    argparser.add_argument('-l', '--keep-split-antecedents', action='store_true', default=False,
                           help='evaluate also split-antecedents; ignored otherwise')
    argparser.add_argument('-d', '--evaluate-discourse-deixis', action='store_true', default=False,
                           help='evaluate discourse deixis instead of identity anaphora')
    argparser.add_argument('-p', '--partial-match', action='store_true', default=False,
                           help='use partial match for matching key and system mentions; exact match otherwise')
    argparser.add_argument('--partial-match-method', choices=['default', 'craft'], default='default',
                           help='the method used for partial matching')
    argparser.add_argument('--only-split-antecedent', action='store_true', default=False,
                           help='report F1 scores on split antecedent alignments')
    argparser.add_argument('--allow-boundary-crossing', action='store_true', default=False,
                           help='to allow partial boundary overlapping')
    argparser.add_argument('--np-only', action='store_true', default=False, help='evaluate only NP metnions')
    argparser.add_argument('--remove-nested-mentions', action='store_true', default=False,
                           help='evaluate only flat metnions')
    argparser.add_argument('-t','--shared-task',
                           choices=['conll12', 'crac18', 'craft19', 'crac22', 'codicrac22ar', 'codicrac22br',
                                    'codicrac22dd'],
                           help='use specific shared task settings, this will overridde all other settings, for more detail please check shared task website')

    args = vars(argparser.parse_args())

    metric_dict = {
        'muc': evaluator.muc, 'bcub': evaluator.b_cubed,
        'ceafe': evaluator.ceafe, 'ceafm': evaluator.ceafm,
        'blanc': [evaluator.blancc, evaluator.blancn], 'lea': evaluator.lea,
        'mention': (evaluator.mention_overlap if args['partial_match'] else evaluator.mentions),
        # we can use mention_overlap for partial match and mention F1 for exact_match
        'zero': evaluator.als_zeros,
        'non-referring': evaluator.evaluate_non_referrings,
        'bridging': evaluator.evaluate_bridgings
    }
    if args['shared_task']:
        key_file = args['key_file']
        sys_file = args['sys_file']
        args = SHARED_TASK_SETTINGS[args['shared_task']]
        args['key_file'] = key_file
        args['sys_file'] = sys_file
    else:
        if 'all' in args['metrics']:
            if args['format'] == 'conll':
                args['metrics'] = [m for m in metric_dict.keys() if m not in ['zero', 'non-referring', 'bridging']]
            elif args['format'] == 'corefud':
                args['metrics'] = [m for m in metric_dict.keys() if m not in ['non-referring', 'bridging']]
            else:
                args['metrics'] = [m for m in metric_dict.keys() if m not in ['zero']]
        elif 'conll' in args['metrics']:
            args['metrics'] = ['muc', 'bcub', 'ceafe']

        if args['only_split_antecedent']:
            for must_true in ['keep_split_antecedent', 'keep_singletons']:
                if args[must_true] == False:
                    autoreset_msg(must_true, True, 'only_split_antecedent')
                    args[must_true] = True
            for un_metric in ['bridging', 'non-referring', 'zero']:
                if un_metric in args['metrics']:
                    metric_autoremove_msg(un_metric, 'only_split_antecedent')

        if args['evaluate_discourse_deixis']:
            for must_true in ['keep_split_antecedent', 'keep_singletons']:
                if args[must_true] == False:
                    autoreset_msg(must_true, True, 'evaluate_discourse_deixis')
                    args[must_true] = True

            for must_false in ['only_split_antecedent']:
                if args[must_false] == True:
                    autoreset_msg(must_false, False, 'evaluate_discourse_deixis')
                    args[must_false] = False

            for un_metric in ['bridging', 'non-referring', 'zero']:
                if un_metric in args['metrics']:
                    metric_autoremove_msg(un_metric, 'evaluate_discourse_deixis')

    compatibility_check(args)

    args['metrics'] = [(name, metric_dict[name]) for name in args['metrics']]

    coref_metrics = ['muc', 'bcub', 'ceafe', 'ceafm', 'blanc', 'lea', 'mention', 'zero']
    has_coref_metrics = any([m in args['metrics'] for m in coref_metrics])

    args['keep_non_referring'] = 'non-referring' in args['metrics']
    args['keep_bridging'] = 'bridging' in args['metrics']

    msg = "The scorer is evaluating "
    if args['evaluate_discourse_deixis']:
        msg += 'discourse deixis'
    elif args['only_split_antecedent']:
        msg += 'only split-antecedents'
    elif not has_coref_metrics:
        if args['keep_non_referring']:
            msg += 'non-referring mentions'
        if args['keep_bridging']:
            msg += ', ' if msg[-1] == ' ' else ''
            msg += 'bridging relations'
    else:
        msg += 'corferent markables'
        if args['keep_singletons']:
            msg += ', singletons'
        if args['keep_split_antecedent']:
            msg += ', split-antecedents'
        if args['keep_non_referring']:
            msg += ', non-referring mentions'
        if args['keep_bridging']:
            msg += ', bridging relations'
        if args['np_only']:
            msg += ', keep only np mentions'
        if args['remove_nested_mentions']:
            msg += ', excluding nested mentions'

    msg += " using {:s} match evaluation setting{:s}.\n".format('partial' if args['partial_match'] else 'exact',
                                                                ' with {:s} method'.format(
                                                                    args['partial_match_method']) if args[
                                                                    'partial_match'] else '')
    msg += "The following metrics will be evaluated: {:s}\n".format(", ".join([name for name, _ in args['metrics']]))
    print(msg)

    evaluate(args)


def evaluate(args):
    key_file = args['key_file']
    sys_file = args['sys_file']
    reader = None
    if args['format'] == 'ua':
        reader = UAReader(**args)
    elif args['format'] == 'corefud':
        reader = CorefUDReader(**args)
    else:
        reader == CoNLLReader(**args)

    reader.get_coref_infos(key_file, sys_file)

    conll = 0
    conll_subparts_num = 0

    for name, metric in args['metrics']:
        if name == 'non-referring':
            recall, precision, f1 = evaluate_non_referrings(
                reader.doc_non_referring_infos)
            print('============================================')
            print('Non-referring markable identification scores:')
            print('Recall: %.2f' % (recall * 100),
                  ' Precision: %.2f' % (precision * 100),
                  ' F1: %.2f' % (f1 * 100))
        elif name == 'bridging':
            score_ar, score_fbm, score_fbe = evaluator.evaluate_bridgings(reader.doc_bridging_infos)
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
        else:
            recall, precision, f1 = evaluator.evaluate_documents(
                reader.doc_discourse_deixis_infos if args['evaluate_discourse_deixis'] else reader.doc_coref_infos,
                metric,
                beta=1,
                only_split_antecedent=args['only_split_antecedent'])
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


main()
