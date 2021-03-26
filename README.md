# The Universal Anaphora Scorer

## About

The Universal Anaphora (UA) scorer is a Python scorer for the varieties of anaphoric reference covered by the Universal Anaphora guidelines, which include identity reference, bridging reference, and discourse deixis:

https://github.com/UniversalAnaphora

The scorer builds on the original reference Coreference scorer [Pradhan et al, 2014] developed for scoring the CoNLL 2011 and 2012 shared tasks using the OntoNotes corpus [Pradhan et al, 2011; Pradhan et al, 2012]:

https://github.com/conll/reference-coreference-scorers

and its reimplementation in Python by Moosavi, also extended to compute the LEA score [Moosavi and Strube, 2016] and to evaluate non-referring expressions evaluation and cover singletons [Poesio et al, 2018]

https://github.com/ns-moosavi/LEA-coreference-scorer


## Usage

The following command evaluates coreference outputs related to the UA format:

`python ua-scorer.py key system [options]`

where `key` and `system` are the location of the key (gold) and system (predicted) files.


## Evaluation Metrics

The above command reports MUC [Vilain et al, 1995], B-cubed [Bagga and Baldwin, 1998], CEAF [Luo et al., 2005], BLANC [Recasens and Hovy, 2011], LEA [Moosavi and Strube, 2016] and the averaged CoNLL score (the average of the F1 values of MUC, B-cubed and CEAFe) [Denis and Baldridge, 2009a; Pradhan et al., 2014].

You can also only select specific metrics by including one or some of the `muc`, `bcub`, `ceafe`, `ceafm`, `blanc` and `lea` options in the input arguments.
For instance, the following command only reports the CEAFe and LEA scores:

`python ua-scorer.py key system ceafe lea`

The first and second arguments after `ua-scorer.py` have to be 'key' and 'system', respectively. The order of the other options is arbitrary.

## Evaluation Modes

Apart from coreference relations, the UA dataset also contains annotations for singletons, non-referring, split-antecedent, bridging and discourse-deixis markables.

Non-referring markables are annotated in the `Identity` column with EntityIDs that end with the `-Pseudo` tag so that they can be distinguished from referring markables (coreferent markables or singletons). Split-antecedents are also marked in the Identity column using the `Element-of` feature in the antecedents.
Bridging references and discourse-deixis are annotated in separate columns.
After extracting all markables, all markables whose corresponding coreference chain is of size one, are specified as singletons.
By distinguishing coreferent markables, singletons, bridging, split-antecedent and non-referring markables, we can perform coreference evaluations in various settings by using the following two options:

1) `remove_singletons`:  if this option is included in the command, all singletons in the key and system files will be excluded from the evaluation.

2) `remove_split_antecedent`: if this option is included in the command, all split-antecedent in the key and system files will be excluded from the evaluation.

3) `keep_non_referring`:  if this option is included in the command, all markables that are annotated as non_referring, both in the key and system files, will be included in the evaluation.
If this option is included, separate recall, precision, and F1 scores would be reported for identifying the annotated non-referring markables.

4) `keep_bridging`: if this option is included in the command, all markables that are annotated as bridging, both in the key and system files, will be included in the evaluation.
If this option is included, separate recall, precision, and F1 scores would be reported for identifying the annotated bridging markables.

5) `evaluate_discourse_deixis`: if this option is included in the command the scorer will only evaluate discourse deixis using the metrics specified.

6) `only_split_antecedent`: there is also an option to only assess the quality of the alignment between the split-antecedents in the key and system.

As a result, if you only run `python arrau-scorer.py key system` without any additional option, the evaluation is performed by incorporating all coreferent, split-antecedent and singleton markables and without considering non-referring, bridging or discourse-deixis markables.

Overall, the above options enable the following evaluation modes:

## Evaluating coreference relations only

This evaluation mode is compatible with the coreference evaluation of the OntoNotes dataset in which only coreferring markables are evaluated.

To do so, the `remove_singletons` option should be included in the evaluation command:

`python ua-scorer.py key system remove_singletons remove_split_antecedent`

In this mode, all split-antecedent, singletons, bridging, discourse-deixis and non-referring mentions will be skipped from coreference evaluations.

## Evaluating coreference relations (include split-antecedents) and singletons

This is the default evaluation mode of the UA dataset and its corresponding command is

`python ua-scorer.py key system`

In this mode, both coreferring markables, split-antecedents and singletons are evaluated by the specified evaluation metrics.
Apart from the MUC metric, all other evaluation metrics handle singletons.
The only case in which MUC handles singletons is when they are incorrectly included in system detected coreference chains. In this case, MUC penalizes the output for including additional incorrect coreference relations. Otherwise, MUC does not handle, or in other words skip, markables that do not have coreference links.
You can refer to Pradhan et al [2014] and Moosavi and Strube [2016] for more details about various evaluation metrics.

## Evaluating all markables (exclude discourse-deixis)

In this evaluation setting, all specified markables including coreference, split-antecedents, singleton, bridging and non-referring mentions are taken in to account for evaluation.
The scores of coreference evaluation metrics would be the same as those of the above mode, i.e. evaluating coreference relations, split-antecedents and singletons.
However, separate scores would be reported for identifying the annotated non-referring markables and bridging relations.

The following command performs coreference evaluation using this mode

`python ua-scorer.py key system keep_non_referring keep_bridging`

## Evaluating discourse-deixis

In this evaluation setting only discourse-deixis will be evaluated.

The following command performs coreference evaluation using this mode

`python ua-scorer.py key system evaluate_discourse_deixis`

## Evaluating only split-antecedent alignment

In this evaluation setting only split-antecedents will be evaluated, the scores are reported according to the alignment between split-antecedents in the key and system predictions.

The following command performs coreference evaluation using this mode

`python ua-scorer.py key system only_split_antecedent`

## Minimum Span Evaluation

The UA dataset may contain a MIN attribute which indicates the minimum string that a coreference
resolver must identify for the corresponding markable.
For minimum span evaluation, a system detected boundary for a markable is considered as correct if it contains the MIN string and doesn't go beyond the annotated maximum boundary.

To perform minimum span evaluations, add one of the `MIN`, `min` or `min_spans` options to the input arguments.
For instance, the following command reports all standard evaluation metrics using minimum spans to specify markables instead of maximum spans:

`python ua-scorer.py key system min`

## Authors

* Juntao Yu, Queen Mary University of London, juntao.cn@gmail.com
* Nafise Moosavi, UKP, TU Darmstadt, ns.moosavi@gmail.com
* Silviu Paun, Queen Mary University of London, spaun3691@gmail.com
* Massimo Poesio, Queen Mary University of London, poesio@gmail.com

The original Reference Coreference Scorer was developed by:

*  Emili Sapena, Universitat Politècnica de Catalunya, http://www.lsi.upc.edu/~esapena, esapena@lsi.upc.edu
*  Sameer Pradhan, https://cemantix.org, pradhan@cemantix.org
*  Sebastian Martschat, sebastian.martschat@h-its.org
*  Xiaoqiang Luo, xql@google.com

## References

  Massimo Poesio, Yulia Grishina, Varada Kolhatkar, Nafise  Moosavi, Ina  Roesiger, Adam  Roussel, Fabian Simonjetz, Alexandra Uma, Olga Uryupina, Juntao Yu, and Heike Zinsmeister. 2018.
  Anaphora resolution with the ARRAU corpus.
  In Proc. of the NAACL Worskhop on Computational Models of Reference, Anaphora and Coreference (CRAC), pages 11–22, New Orleans.

  Nafise Sadat Moosavi and Michael Strube. 2016.
  Which Coreference Evaluation Metric Do You Trust? A Proposal for a Link-based Entity Aware Metric.
  In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics.

  Sameer Pradhan, Xiaoqiang Luo, Marta Recasens, Eduard Hovy, Vincent Ng, and Michael Strube. 2014.
  Scoring coreference partitions of predicted mentions: A reference implementation.
  In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),
  Baltimore, Md., 22–27 June 2014, pages 30–35.

  Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, Yuchen Zhang.   2012.
  CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes
  In Proceedings of the Joint Conference on EMNLP and CoNLL: Shared Task, pages 1-40

  Marta Recasens and Eduard Hovy.  2011.
  BLANC: Implementing the Rand Index for coreference evaluation.
  Natural Language Engineering, 17(4):485–510.

  Sameer Pradhan, Lance Ramshaw, Mitchell Marcus, Martha Palmer, Ralph Weischedel, and Nianwen Xue.  2011.
  CoNLL-2011 shared task: Modeling unrestricted coreference in OntoNotes.
  In Proceedings of CoNLL: Shared Task, pages 1–27.

  Pascal Denis and Jason Baldridge.  2009.
  Global joint models for coreference resolution and named entity classification.
  Procesamiento del Lenguaje Natural, (42):87–96.

  Xiaoqiang Luo. 2005.
  On coreference resolution performance metrics.
  In Proceedings of HLT-EMNLP, pages 25–32.

  Amit Bagga and Breck Baldwin.  1998.
  Algorithms for scoring coreference chains.
  In Proceedings of LREC, pages 563–566.

  Marc Vilain, John Burger, John Aberdeen, Dennis Connolly, and Lynette Hirschman. 1995.
  A model theoretic coreference scoring scheme.
  In Proceedings of the 6th Message Understanding Conference, pages 45–52.


## Acknowledgments

The development of the scorer was in part supported by the DALI project

http://www.dali-ambiguity.org

DALI is funded by the European Research Council.

