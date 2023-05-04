# The Universal Anaphora Scorer
## Introduction
This repository contains code introduced in the following paper:

**[The Universal Anaphora Scorer](https://aclanthology.org/2022.lrec-1.521/)**  
Juntao Yu, Sopan Khosla, Nafise Moosavi, Silviu Paun,  Sameer Pradhan and Massimo Poesio  
In *Proceedings of the 13th Language Resources and Evaluation Conference (LREC)*, 2022

**[Scoring Coreference Chains with Split-Antecedent Anaphors](https://arxiv.org/abs/2205.12323)**  
Silviu Paun*, Juntao Yu*, Nafise Moosavi and Massimo Poesio  `*equal contribution`  
In *Arxiv.org*, 2022 

**[The CODI-CRAC 2021 Shared Task on Anaphora, Bridging, and Discourse Deixis in Dialogue](https://aclanthology.org/2021.codi-sharedtask.1/)**
Sopan Khosla, Juntao Yu, Ramesh Manuvinakurike, Vincent Ng, Massimo Poesio, Michael Strube, Carolyn Rosé
In *Proceedings of the CODI-CRAC 2021 Shared Task on Anaphora, Bridging, and Discourse Deixis in Dialogue (CODI-CRAC@EMNLP)*, 2021

**[The CODI-CRAC 2022 Shared Task on Anaphora, Bridging, and Discourse Deixis in Dialogue](https://aclanthology.org/2022.codi-crac.1/)**
Juntao Yu, Sopan Khosla, Ramesh Manuvinakurike, Lori Levin, Vincent Ng, Massimo Poesio, Michael Strube, Carolyn Rosé
In *Proceedings of the CODI-CRAC 2022 Shared Task on Anaphora, Bridging, and Discourse Deixis in Dialogue (CODI-CRAC@COLING)*, 2022

**[Findings of the Shared Task on Multilingual Coreference Resolution](https://aclanthology.org/2022.crac-mcr.1/)**
Zdeněk Žabokrtský, Miloslav Konopík, Anna Nedoluzhko, Michal Novák, Maciej Ogrodniczuk, Martin Popel, Ondřej Pražák, Jakub Sido, Daniel Zeman, Yilun Zhu
In *Proceedings of the CRAC 2022 Shared Task on Multilingual Coreference Resolution (CRAC@COLING)*, 2022




## About

The Universal Anaphora (UA) scorer is a Python scorer for the varieties of anaphoric reference covered by the Universal Anaphora guidelines, which include identity reference (including singletons, split-antecedent anaphora, zero anaphora, discontinuous mentions, partial mention matching), non-referring expressions, bridging reference, and discourse deixis:

https://github.com/UniversalAnaphora

The scorer builds on the original reference Coreference scorer [Pradhan et al, 2014] developed for scoring the CoNLL 2011 and 2012 shared tasks using the OntoNotes corpus [Pradhan et al, 2011; Pradhan et al, 2012]:

https://github.com/conll/reference-coreference-scorers

and its reimplementation in Python by Moosavi, also extended to compute the LEA score [Moosavi and Strube, 2016] and to evaluate non-referring expressions evaluation and cover singletons [Poesio et al, 2018]

https://github.com/ns-moosavi/LEA-coreference-scorer


## Usage

The following command evaluates coreference outputs:

`python ua-scorer.py key system [options]`

where `key` and `system` are the location of the key (gold) and system (predicted) files.

## Input Formats
The scorer support three input formats (UA-exploded, CorefUD/UA-compact and CoNLL), the CoNLL format only support continuous mentions; the UA-exploded and CorefUD/UA-compact format in addition support discontinuous mentions. For detailed discussion on how discontinuous mentions were presented in those format please follow the link for specific format.  The option `[-f|--format]` can be used to specify the input format:

* `ua` **[default]**: [UA-exploded format](https://github.com/UniversalAnaphora/UniversalAnaphora/blob/main/documents/UA_CONLL_U_Plus_proposal_v1.0.md) 
* `corefud`: [CorefUD/UA-compact format](https://ufal.mff.cuni.cz/~zeman/2022/docs/corefud-1.0-format.pdf)
* `conll`: CoNLL format

## Evaluation Metrics
The scorer support all the major corefernce metrics as well as scores commonly used by other anaphora relations (e.g. bridging, non-referring, discourse deixis or zeros). The option `[-m|--metrics]` can be used to specify the metrics used by the scorer. Here is a list of metrics currently supported by the scorer:
* `muc`: MUC [Vilain et al, 1995]
* `bcub`: B-cubed [Bagga and Baldwin, 1998] 
* `ceafe`: Entity level CEAF [Luo et al., 2005]
* `ceafm`: Mention level CEAF [Luo et al., 2005]
* `blanc`: BLANC [Recasens and Hovy, 2011] 
* `lea`: LEA [Moosavi and Strube, 2016]
* `conll` **[default]**: the averaged CoNLL score (the average F1 of MUC, B-cubed and CEAFe) [Denis and Baldridge, 2009a; Pradhan et al., 2014]
* `mention`: the mention F1 score when exact matching is used on mentions
* `mor`: average mention overlap ratio when partial matching is used [Žabokrtský et al., 2022]
* `zero`: (only for `corefud` format) the application-related coreference scores [Tuggener, 2014] for zero anaphors [Žabokrtský et al., 2022]
* `non-referring`: (only for `ua` format) the F1 score for non-referring expressions
* `bridging`: (only for `ua` format) the entity-based F1, mention-based F1 and bridging anaphora recognition F1 scores.
* `all`: report all scores supported for specified format

For instance, the following command reports the CEAFe and LEA scores:

`python ua-scorer.py key system -m ceafe lea`

## Mention Matching

The UA and Corefud dataset may contain a MIN/head attribute which indicates the minimum string that a coreference
resolver must identify for the corresponding markable.
The minimum span are used to allow partial/fuzzy mention matching, the scorer support three mention matching method (exact match and two partial match methods):

### Exact Matching
By default, the scorer uses exact mention matching. In exact matching, the two mentions are considered matching if and only if they consist of the same set of words. A word is defined here only by its position within the sentence and by position of the sentence within the whole file.

### Partial Matching
The partial mention matching can be triggered by using the `[-p|--partial-match]` options and with additional option `[--partial-match-method]` to specify the method used for partial matching, there are two options for partial matching:

#### Default:
In default partial matching method, a system detected boundary for 
a mention is considered as correct if it contains the MIN/head string 
and doesn't go beyond the annotated maximum boundary. To align the mentions
in the key and response:
* We first align the mentions based on the exact matching to exclude them from partial matching step. 
* To align the remaining mentions, we first compute the recall (the precision will always be 100% according to our definition of partial matching) between all the mention pairs between key and response
* The recalls are then used with the Kuhn-Munkres algorithm Kuhn (1955); Munkres (1957) to find the best alignment between those mentions.

To perform evaluation with partial mention matching using our default method, add one `-p` or `--partial-match` options to the input arguments.
For instance, the following command reports the CoNLL evaluation metrics using default partial mention method:

`python ua-scorer.py key system -p` 

### Craft:
The scorer has an option to allow use the partial mention matching method defined in CRAFT 2019 shared task (Baumgartner et al., 2019). The method considers 
a predicted mention correct if any continuous span of the predicted mention overlaps with and doesn't go beyond the first span of the key mention. To perform evaluation 
with partial mention matching using CRAFT method:

`python ua-scorer.py key system -p --partial-match-method craft`

## Zeros
The scorer has the option to allow include zeros in the evaluation, the option is supported by both the `ua` and `corefud` formats. To include zeros in your evaluation, add `-z` or `--keep-zeros` options to the input arguments:

`python ua-scorer.py key system -z`

Currently, the scorer aligns the zeros in the key and system using their positions (`linear`), assuming there is a consistent annotation guideline on where the zeros should be positioned. A linguistically motivated method (`dependent`) based on zero's dependence is under development for the `corefud` format to use parse trees to align the zeros in a different way. The `--zero-match-metod` option is used to specify the method for the alignments, by default it uses `linear` method.


## Other Options
Apart from metrics, partial mention matching, zeros, the scorer has the following additional options:
* `-s|--keep-singletons`:  if this option is included in the command, all singletons in the key and system files will be included in the evaluation.
* `-l|--keep-split-antecedents`: (only for `ua` format) if this option is included in the command, all split-antecedent in the key and system files will be included in the evaluation.
* `-d|--evaluate-discourse-deixis`: (only for `ua` format) if this option is included in the command the scorer will only evaluate discourse deixis using the metrics specified.
* `--only-split-antecedent`: (only for `ua` format) there is also an option to only assess the quality of the alignment between the split-antecedents in the key and system.
* `--allow-boundary-crossing`: (only for `ua` format) this is used when partial mention overlapping (e.g. (a (b a) b) ) is allowed in the corpus you need include the markable_id in the close bracket for the mention
* `--np-only`: (only for `conll` format) evaluate only NP mentions
* `--remove-nested-mentions`: (only for `conll` format) evaluate only non-nested mentions


## Evaluation Modes
As a result, if you only run `python ua-scorer.py key system` without any additional option, the evaluation is performed by reporting MUC, B-cubed, CEAFe and CoNLL F1 scores on coreference clusters using UA-exploded format and exact mention matching but without considering split-antecedent, singleton, non-referring, bridging or discourse-deixis markables.

Overall, the above options enable the following evaluation modes: 

### Evaluating coreference relations only (e.g. CoNLL 2012)

This evaluation mode is compatible with the coreference evaluation of the OntoNotes dataset in which only coreferring markables are evaluated.


`python ua-scorer.py key system -f conll` 

it can be also called by using the `[-t|--shared-task]` option:

`python ua-scorer.py key system -t conll12`


### Evaluating coreference relations (include split-antecedents) and singletons (e.g. CODI-CRAC 2021, 2022 Task 1)

This evaluation mode are only available for the `ua` format and its corresponding command is:

`python ua-scorer.py key system -sl `

or with `[-t|--shared-task]` option:

`python ua-scorer.py key system -t codicrac22ar`


In this mode, both coreferring markables, split-antecedents and singletons are evaluated by the specified evaluation metrics.
Apart from the MUC metric, all other evaluation metrics handle singletons.
The only case in which MUC handles singletons is when they are incorrectly included in system detected coreference chains. In this case, MUC penalizes the output for including additional incorrect coreference relations. Otherwise, MUC does not handle, or in other words skip, markables that do not have coreference links.
You can refer to Pradhan et al [2014] and Moosavi and Strube [2016] for more details about various evaluation metrics.

### Evaluating coreference relatons, singletons and zeros using partial mention matching (e.g. CRAC 2022)
This evaluation mode is used by the CRAC 2022 shared task, it includes coreference relations, singletons, zeros (with gold zeros provided) but not split-antecedents. It reports MUC, B-cubed, CEAFe, CoNLL F1, mention overlapping scores and anaphora level scores for zero. Its corresponding command is:  

`python ua-scorer.py key system -f corefud -sz -m muc bcub ceafe mention zero -p`

or using the `[-t|--shared-task]` shortcuts:

`python ua-scorer.py key system -t crac22`

### Evaluating coreference relations, singletons using CRAFT partial mention matching (e.g. CRAFT 2019)
This evaluation mode is used by the CRAFT 2019 shared task, it includes coreference relations, singletons. It reports MUC, B-cubed, CEAFe, CoNLL F1. Its corresponding command is:  

`python ua-scorer.py key system -s -m muc bcub ceafe -p --partial-match-method craft`

or using the `[-t|--shared-task]` shortcuts:

`python ua-scorer.py key system -t craft19`


### Evaluating coreference realations, singletons and non-referrings (e.g. CRAC 2018)

In this evaluation setting, coreference relations, singletons and non-referring mentions are taken into account for evaluation.
The coreference relations and singletons are evaluated using the coreference metrics while the non-referring expressions are scored separately.

The following command performs evaluation using this mode:

`python ua-scorer.py key system -s -m muc bcub ceafe non-referring`

or using the `[-t|--shared-task]` shortcuts:

`python ua-scorer.py key system -t crac18`

### Evaluating Bridging reference (e.g. CODI-CRAC 2021, 2022 Task 2)
In this evaluation setting only bridging reference will be evaluated.

The following command performs evaluation using this mode

`python ua-scorer.py key system -m bridging` or

`python ua-scorer.py key system -t codicrac22br`

### Evaluating discourse-deixis (e.g. CODI-CRAC 2021, 2022 Task 3)

In this evaluation setting only discourse-deixis will be evaluated.

The following command performs evaluation using this mode

`python ua-scorer.py key system -d` or

`python ua-scorer.py key system -t codicrac22dd`




## Authors

* Juntao Yu, Queen Mary University of London, juntao.cn@gmail.com
* Nafise Moosavi, UKP, TU Darmstadt, ns.moosavi@gmail.com
* Silviu Paun, Queen Mary University of London, spaun3691@gmail.com
* Massimo Poesio, Queen Mary University of London, poesio@gmail.com
* Michal Novák, Charles University, Prague, Czech Republic, mnovak@ufal.mff.cuni.cz 
* Yilun Zhu, Georgetown University, Washington D.C., USA, yz565@georgetown.edu 
* Martin Popel, Charles University, Prague, Czech Republic, popel@ufal.mff.cuni.cz

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

  Zdeněk Žabokrtský, Miloslav Konopík, Anna Nedoluzhko, Michal Novák, Maciej Ogrodniczuk, Martin Popel, Ondřej Pražák, Jakub Sido, Daniel Zeman, and Yilun Zhu. 2022.
  Findings of the Shared Task on Multilingual Coreference Resolution.
  In Proceedings of the CRAC 2022 Shared Task on Multilingual Coreference Resolution, pages 1–17.

  Don Tuggener. 2014.
  Coreference Resolution Evaluation for Higher Level Applications.
  In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics, volume 2: Short Papers, pages 231–235.

## Acknowledgments

The development of the scorer was in part supported by the DALI project

http://www.dali-ambiguity.org

DALI is funded by the European Research Council.

