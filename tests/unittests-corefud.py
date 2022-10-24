import sys
sys.path.insert(1, '..')
from pytest import approx
from scorer.corefud.reader import CorefUDReader
from scorer.eval.evaluator import evaluate_documents as evaluate
from scorer.eval.evaluator import muc, b_cubed, ceafe, lea, ceafm, blancc, blancn, mention_overlap

TOL = 1e-4


def read(key, response, exact_match=False):
  args = {
    "format": 'corefud',
    "keep_singletons": True,
    "partial_match":not exact_match
  }
  reader = CorefUDReader(**args)
  folder = 'partial-match-corefud' if '/' in key else 'original-conll'
  reader.get_coref_infos('tests-corefud/%s/%s' % (folder,key), 'tests-corefud/%s/%s' % (folder, response))
  return reader.doc_coref_infos

def test_A1():
  doc = read('TC-A.key', 'TC-A-1.response', exact_match=True)
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, ceafe) == (1, 1, 1)
  assert evaluate(doc, ceafm) == (1, 1, 1)
  assert evaluate(doc, lea) == (1, 1, 1)
  assert evaluate(doc, [blancc,blancn]) == (1,1,1)


def test_A2():
  doc = read('TC-A.key', 'TC-A-2.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1 / 3, 1 / 1, 1 / 2])
  assert evaluate(doc, b_cubed) == approx([(7 / 3) / 6, 3 / 3, 14 / 25])
  assert evaluate(doc, ceafe) == approx([0.6, 0.9, 0.72])
  assert evaluate(doc, ceafm) == approx([0.5, 1, 0.66667], abs=TOL)
  assert evaluate(doc, lea) == approx([(1 + 3 * (1 / 3)) / 6, 1, 0.5])
  assert evaluate(doc, [blancc,blancn]) == approx([0.21591, 1, 0.35385], abs=TOL)

def test_A3():
  doc = read('TC-A.key', 'TC-A-3.response', exact_match=True)
  assert evaluate(doc, muc) == approx([3 / 3, 3 / 5, 0.75])
  assert evaluate(doc,
      b_cubed) == approx([6 / 6, (4 + 7 / 12) / 9, 110 / 163])
  assert evaluate(doc, ceafe) == approx([0.88571, 0.66429, 0.75918], abs=TOL)
  assert evaluate(doc, lea) == approx([
      1, (1 + 3 * (1 / 3) + 4 * (3 / 6)) / 9,
      2 * (1 + 3 * (1 / 3) + 4
        * (3 / 6)) / 9 / (1 + (1 + 3 * (1 / 3) + 4 * (3 / 6)) / 9)
  ])
  assert evaluate(doc, ceafm) == approx([1, 0.66667, 0.8], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([1, 0.42593, 0.59717], abs=TOL)

def test_A4():
  doc = read('TC-A.key', 'TC-A-4.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1 / 3, 1 / 3, 1 / 3])
  assert evaluate(doc, b_cubed) == approx([
      (3 + 1 / 3) / 6, (1 + 4 / 3 + 1 / 2) / 7,
      2 * (5 / 9) * (17 / 42) / ((5 / 9) + (17 / 42))
  ])
  assert evaluate(doc, ceafe) == approx([0.73333, 0.55, 0.62857], abs=TOL)
  assert evaluate(doc, lea) == approx([(1 + 2 + 0) / 6,
      (1 + 3 * (1 / 3) + 2 * 0 + 0) / 7,
      2 * 0.5 * 2 / 7 / (0.5 + 2 / 7)])
  assert evaluate(doc, ceafm) == approx([0.66667, 0.57143, 0.61538], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.35227, 0.27206, 0.30357], abs=TOL)

def test_A5():
  doc = read('TC-A.key', 'TC-A-5.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1 / 3, 1 / 4, 2 / 7])
  assert evaluate(doc, b_cubed) == approx([
      (3 + 1 / 3) / 6, 2.5 / 8,
      2 * (5 / 9) * (5 / 16) / ((5 / 9) + (5 / 16))
  ])
  assert evaluate(doc, ceafe) == approx([0.68889, 0.51667, 0.59048], abs=TOL)
  assert evaluate(doc,
      lea) == approx([(1 + 2 + 3 * 0) / 6,
      (1 + 4 * (1 / 6) + 2 * 0 + 1 * 0) / 8,
      2 * 0.5 * (5 / 24) / (0.5 + (5 / 24))])
  assert evaluate(doc, ceafm) == approx([0.66667, 0.5, 0.57143], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.35227, 0.19048, 0.24716], abs=TOL)


def test_A6():
  doc = read('TC-A.key', 'TC-A-6.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1 / 3, 1 / 4, 2 / 7])
  assert evaluate(doc, b_cubed) == approx([
      (10 / 3) / 6, (1 + 4 / 3 + 1 / 2) / 8,
      2 * (5 / 9) * (17 / 48) / ((5 / 9) + (17 / 48))
  ])
  assert evaluate(doc, ceafe) == approx([0.73333, 0.55, 0.62857], abs=TOL)
  assert evaluate(doc, lea) == approx([(1 + 2 + 3 * 0) / 6,
      (1 + 3 / 3 + 2 * 0 + 2 * 0) / 8,
      2 * 0.5 * 1 / 4 / (0.5 + 1 / 4)])
  assert evaluate(doc, ceafm) == approx([0.66667, 0.5, 0.57143], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.35227, 0.20870, 0.25817], abs=TOL)


def test_A7():
  doc = read('TC-A.key', 'TC-A-7.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1 / 3, 1 / 3, 1 / 3])
  assert evaluate(doc, b_cubed) == approx([
      (10 / 3) / 6, (1 + 4 / 3 + 1 / 2) / 7,
      2 * (5 / 9) * (17 / 42) / ((5 / 9) + (17 / 42))
  ])
  assert evaluate(doc, ceafe) == approx([0.73333, 0.55, 0.62857], abs=TOL)
  assert evaluate(doc, lea) == approx([(1 + 2 + 3 * 0) / 6,
      (1 + 3 / 3 + 2 * 0 + 1 * 0) / 7,
      2 * 0.5 * 2 / 7 / (0.5 + 2 / 7)])
  assert evaluate(doc, ceafm) == approx([0.66667, 0.57143, 0.61538], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.35227, 0.27206, 0.30357], abs=TOL)


def test_A10():
  doc = read('TC-A.key', 'TC-A-10.response', exact_match=True)
  assert evaluate(doc, muc) == approx([0, 0, 0])
  assert evaluate(doc, b_cubed) == approx([3 / 6, 6 / 6, 2 / 3])
  assert evaluate(doc, lea) == approx(
      [1 / 6, 1 / 6, 2 * 1 / 6 * 1 / 6 / (1 / 6 + 1 / 6)])
  assert evaluate(doc, [blancc, blancn]) == approx([0.5, 0.36667, 0.42308], abs=TOL)


def test_A11():
  doc = read('TC-A.key', 'TC-A-11.response', exact_match=True)
  assert evaluate(doc, muc) == approx([3 / 3, 3 / 5, 6 / 8])
  assert evaluate(doc, b_cubed) == approx(
      [6 / 6, (1 / 6 + 2 * 2 / 6 + 3 * 3 / 6) / 6, 14 / 25])
  assert evaluate(doc,
      lea) == approx([(0 + 2 + 3) / 6, 4 / 15,
      2 * 5 / 6 * 4 / 15 / (5 / 6 + 4 / 15)])
  assert evaluate(doc, [blancc, blancn]) == approx([0.5, 0.13333, 0.21053], abs=TOL)

def test_A12():
  doc = read('TC-A.key', 'TC-A-12.response', exact_match=True)
  assert evaluate(doc, muc) == approx([0, 0, 0])
  assert evaluate(doc, b_cubed) == approx([
      (1 + 1 / 2 + 2 / 3) / 6, 4 / 7,
      2 * (13 / 36) * (4 / 7) / ((13 / 36) + (4 / 7))
  ])
  assert evaluate(doc, lea) == approx(
      [1 / 6, 1 / 7, 2 * 1 / 6 * 1 / 7 / (1 / 6 + 1 / 7)])
  assert evaluate(doc, [blancc, blancn]) == approx([0.22727, 0.11905, 0.15625], abs=TOL)

def test_A13():
  doc = read('TC-A.key', 'TC-A-13.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1 / 3, 1 / 6, 2 / 9])
  assert evaluate(doc, b_cubed) == approx([
      (1 + 1 / 2 + 2 * 2 / 3) / 6, (1 / 7 + 1 / 7 + 2 * 2 / 7) / 7,
      2 * (17 / 36) * (6 / 49) / ((17 / 36) + (6 / 49))
  ])
  assert evaluate(doc,
      lea) == approx([(1 * 0 + 2 * 0 + 3 / 3) / 6, 1 / 21,
      2 * 1 / 6 * 1 / 21 / (1 / 6 + 1 / 21)])
  assert evaluate(doc, [blancc, blancn]) == approx([0.125, 0.02381, 0.04], abs=TOL)

def test_B1():
  doc = read('TC-B.key', 'TC-B-1.response', exact_match=True)
  assert evaluate(doc, lea) == approx([(2 * 0 + 3 / 3) / 5, (3 * 0 + 2) / 5,
      2 * 1 / 5 * 2 / 5 / (1 / 5 + 2 / 5)])
  assert evaluate(doc, [blancc, blancn]) == approx([1/2 * (1/4 + 1/3), 1/2 * (1/4 + 1/3), 1/2 * (1/4 + 1/3)])

def test_C1():
  doc = read('TC-C.key', 'TC-C-1.response', exact_match=True)
  assert evaluate(doc, lea) == approx([(2 * 0 + 3 / 3 + 2) / 7,
      (3 * 0 + 2 + 2) / 7,
      2 * 3 / 7 * 4 / 7 / (3 / 7 + 4 / 7)])
  assert evaluate(doc, [blancc, blancn]) == approx([1/2 * (2/5 + 10/16), 1/2 * (2/5 + 10/16), 1/2 * (2/5 + 10/16)])


def test_D1():
  doc = read('TC-D.key', 'TC-D-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx(
      [9 / 9, 9 / 10, 2 * (9 / 9) * (9 / 10) / (9 / 9 + 9 / 10)])
  assert evaluate(doc, b_cubed) == approx([
      12 / 12, 16 / 21, 2 * (12 / 12) * (16 / 21) / (12 / 12 + 16 / 21)
  ])
  assert evaluate(doc, lea) == approx([
      (5 + 2 + 5) / 12, (5 + 7 * (11 / 21)) / 12,
      2 * 1 * (5 + 77 / 21) / 12 / (1 + ((5 + 77 / 21) / 12))
  ])

def test_E1():
  doc = read('TC-E.key', 'TC-E-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx(
      [9 / 9, 9 / 10, 2 * (9 / 9) * (9 / 10) / (9 / 9 + 9 / 10)])
  assert evaluate(doc, b_cubed) == approx(
      [1, 7 / 12, 2 * 1 * (7 / 12) / (1 + 7 / 12)])
  assert evaluate(doc, lea) == approx([(5 + 2 + 5) / 12,
      (10 * (20 / 45) + 2) / 12,
      2 * 1 * ((10 * (20 / 45) + 2) / 12)
        / (1 + ((10 * (20 / 45) + 2) / 12))])

def test_F1():
  doc = read('TC-F.key', 'TC-F-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx(
      [2 / 3, 2 / 2, 2 * (2 / 3) * (2 / 2) / (2 / 3 + 2 / 2)])
  assert evaluate(doc, lea) == approx(
      [4 * (2 / 6) / 4, (2 + 2) / 4, 2 * 2 / 6 * 1 / (1 + 2 / 6)])

def test_G1():
  doc = read('TC-G.key', 'TC-G-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx(
      [2 / 2, 2 / 3, 2 * (2 / 2) * (2 / 3) / (2 / 2 + 2 / 3)])
  assert evaluate(doc, lea) == approx(
      [1, (4 * 2 / 6) / 4, 2 * 1 * 2 / 6 / (1 + 2 / 6)])


def test_H1():
  doc = read('TC-H.key', 'TC-H-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1, 1, 1])
  assert evaluate(doc, lea) == approx([1, 1, 1])


def test_I1():
  doc = read('TC-I.key', 'TC-I-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx(
      [2 / 3, 2 / 2, 2 * (2 / 3) * (2 / 2) / (2 / 3 + 2 / 2)])
  assert evaluate(doc, lea) == approx(
      [4 * (2 / 6) / 4, (2 + 2) / 4, 2 * 2 / 6 * 1 / (2 / 6 + 1)])



def test_J1():
  doc = read('TC-J.key', 'TC-J-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx(
      [1 / 2, 1 / 1, 2 * (1 / 2) * (1 / 1) / (1 / 2 + 1 / 1)])
  assert evaluate(doc, lea) == approx([(3 * 1 / 3) / 3, 1,
      2 * 1 / 3 / (1 + 1 / 3)])



def test_K1():
  doc = read('TC-K.key', 'TC-K-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx([3 / 6, 3 / 6, 3 / 6])
  assert evaluate(doc,
      lea) == approx([(7 * (1 + 1 + 1) / 21) / 7,
      (3 / 3 + 3 / 3 + 3 / 3) / 9,
      2 * 3 / 21 * 3 / 9 / (3 / 21 + 3 / 9)])



def test_L1():
  doc = read('TC-L.key', 'TC-L-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx(
      [2 / 5, 2 / 4, 2 * (2 / 5) * (2 / 4) / (2 / 5 + 2 / 4)])
  assert evaluate(doc, lea) == approx([
      (3 * 1 / 3 + 4 * 1 / 6) / 7, (2 + 2 * 0 + 3 / 3) / 7,
      2 * (1 + 2 / 3) / 7 * 3 / 7 / (3 / 7 + (1 + 2 / 3) / 7)
  ])



def test_M1():
  doc = read('TC-M.key', 'TC-M-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx([1, 1, 1])
  assert evaluate(doc, b_cubed) == approx([1, 1, 1])
  assert evaluate(doc, ceafe) == approx([1, 1, 1])
  assert evaluate(doc, lea) == approx([1, 1, 1])
  assert evaluate(doc, ceafm) == approx([1, 1, 1])
  assert evaluate(doc, [blancc, blancn]) == approx([1, 1, 1])


def test_M2():
  doc = read('TC-M.key', 'TC-M-2.response', exact_match=True)
  assert evaluate(doc, muc) == approx([0, 0, 0])
  assert evaluate(doc, lea) == approx([0, 0, 0])
  assert evaluate(doc, [blancc, blancn]) == approx([0, 0, 0])

def test_M3():
  doc = read('TC-M.key', 'TC-M-3.response', exact_match=True)
  assert evaluate(doc, lea) == approx([
      6 * (4 / 15) / 6, (2 + 3 + 0) / 6,
      2 * 4 / 15 * 5 / 6 / (4 / 15 + 5 / 6)
  ])
  #the original is wrong as the |N_r| != 0
  #assert evaluate(doc, [blancc, blancn]) == approx([0.26667, 1, 0.42105], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.26667/2, 1/2, 0.42105/2], abs=TOL)


def test_M4():
  doc = read('TC-M.key', 'TC-M-4.response', exact_match=True)
  assert evaluate(doc, lea) == approx([
      6 * (3 / 15) / 6, 6 * (3 / 15) / 6,
      2 * 3 / 15 * 3 / 15 / (3 / 15 + 3 / 15)
  ])
  assert evaluate(doc, [blancc, blancn]) == approx([0.2, 0.2, 0.2])


def test_M5():
  doc = read('TC-M.key', 'TC-M-5.response', exact_match=True)
  assert evaluate(doc, muc) == approx([0, 0, 0])
  assert evaluate(doc, lea) == approx([0, 0, 0])
  assert evaluate(doc, [blancc, blancn]) == approx([0, 0, 0])


def test_M6():
  doc = read('TC-M.key', 'TC-M-6.response', exact_match=True)
  assert evaluate(doc, lea) == approx([
      6 * (1 / 15) / 6, (2 + 3 * 0 + 1 * 0) / 6,
      2 * 1 / 15 * 2 / 6 / (1 / 15 + 2 / 6)
  ])
  # the original is wrong as the |N_r| != 0
  # assert evaluate(doc, [blancc, blancn]) == approx([0.06667, 0.25, 0.10526], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.06667/2, 0.25/2, 0.10526/2], abs=TOL)


def test_N1():
  doc = read('TC-N.key', 'TC-N-1.response', exact_match=True)
  assert evaluate(doc, muc) == approx([0, 0, 0])
  assert evaluate(doc, lea) == approx([1, 1, 1])
  assert evaluate(doc, [blancc, blancn]) == approx([1, 1, 1])


def test_N2():
  doc = read('TC-N.key', 'TC-N-2.response', exact_match=True)
  assert evaluate(doc, muc) == approx([0, 0, 0])
  assert evaluate(doc, lea) == approx([0, 0, 0])
  assert evaluate(doc, [blancc, blancn]) == approx([0, 0, 0])


def test_N3():
  doc = read('TC-N.key', 'TC-N-3.response', exact_match=True)
  assert evaluate(doc, lea) == approx([1 / 6, 1 / 6, 1 / 6])
  # the original is wrong as the |C_r| != 0
  # assert evaluate(doc, [blancc, blancn]) == approx([0.73333, 1, 0.84615], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.73333/2, 1/2, 0.84615/2], abs=TOL)


def test_N4():
  doc = read('TC-N.key', 'TC-N-4.response', exact_match=True)
  assert evaluate(doc, muc) == approx([0, 0, 0])
  assert evaluate(doc, lea) == approx([3 / 6, 3 / 6, 3 / 6])
  assert evaluate(doc, [blancc, blancn]) == approx([0.2, 0.2, 0.2])


def test_N5():
  doc = read('TC-N.key', 'TC-N-5.response', exact_match=True)
  assert evaluate(doc, lea) == approx([0, 0, 0])
  assert evaluate(doc, [blancc, blancn]) == approx([0, 0, 0])


def test_N6():
  doc = read('TC-N.key', 'TC-N-6.response', exact_match=True)
  assert evaluate(doc, lea) == approx([0, 0, 0])
  # the original is wrong as the |C_r| != 0
  # assert evaluate(doc, [blancc, blancn]) == approx([0.13333, 0.18182, 0.15385], abs=TOL)
  assert evaluate(doc, [blancc, blancn]) == approx([0.13333/2, 0.18182/2, 0.15385/2], abs=TOL)

#################################### EXTRA BASIC TESTS ######################################

def test_O1():
  doc = read('TC-O.key', 'TC-O-1.response', exact_match=True)
  assert evaluate(doc, b_cubed) == approx([4/9, 4/15, 1/3], abs=TOL)

############################# TESTS ON NON-CONTIGUOUS MENTIONS ##############################

def test_NCMA1():
  doc = read('noncontig_mentions/TC-NCMA.key', 'noncontig_mentions/TC-NCMA-1.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)

def test_NCMA2():
  doc = read('noncontig_mentions/TC-NCMA.key', 'noncontig_mentions/TC-NCMA-2.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)

def test_NCMA3():
  doc = read('noncontig_mentions/TC-NCMA.key', 'noncontig_mentions/TC-NCMA-3.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMA4():
  doc = read('noncontig_mentions/TC-NCMA.key', 'noncontig_mentions/TC-NCMA-4.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMA5():
  doc = read('noncontig_mentions/TC-NCMA.key', 'noncontig_mentions/TC-NCMA-5.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)

def test_NCMA6():
  doc = read('noncontig_mentions/TC-NCMA.key', 'noncontig_mentions/TC-NCMA-6.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMA7():
  doc = read('noncontig_mentions/TC-NCMA.key', 'noncontig_mentions/TC-NCMA-7.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMB1():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-1.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)

def test_NCMB2():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-2.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)

def test_NCMB3():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-3.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMB4():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-4.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMB5():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-5.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMB6():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-6.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMB7():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-7.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

def test_NCMB8():
  doc = read('noncontig_mentions/TC-NCMB.key', 'noncontig_mentions/TC-NCMB-8.response')
  assert evaluate(doc, muc) == (2/3, 2/3, 2/3)
  assert evaluate(doc, b_cubed) == (7/10, 7/10, 7/10)

############################# TESTS ON OVERLAPPING MENTIONS ##############################

def test_OLMA1():
  doc = read('overlapping_mentions/TC-OLMA.key', 'overlapping_mentions/TC-OLMA-1.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, mention_overlap) == (1, 1, 1)

def test_OLMA2():
  doc = read('overlapping_mentions/TC-OLMA.key', 'overlapping_mentions/TC-OLMA-2.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, mention_overlap) == (9/23, 1, 2*9/23/(1 + 9/23))

def test_OLMA3():
  doc = read('overlapping_mentions/TC-OLMA.key', 'overlapping_mentions/TC-OLMA-3.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, mention_overlap) == (18/23, 1, 2*18/23/(1+18/23))

def test_OLMA4():
  doc = read('overlapping_mentions/TC-OLMA.key', 'overlapping_mentions/TC-OLMA-4.response')
  assert evaluate(doc, muc) == (3/4, 3/4, 3/4)
  assert evaluate(doc, b_cubed) == (17/24, 17/24, 17/24)
  # assert evaluate(doc, mention_overlap) == (12/23, 1, 2*12/23/(1+12/23))

def test_OLMA5():
  doc = read('overlapping_mentions/TC-OLMA.key', 'overlapping_mentions/TC-OLMA-5.response')
  assert evaluate(doc, muc) == (3/4, 3/3, 6/7)
  assert evaluate(doc, b_cubed) == (17/24, 5/5, 34/41)
  assert evaluate(doc, mention_overlap) == (15/23, 1, 2*15/23/(1+15/23))

def test_OLMA6():
  doc = read('overlapping_mentions/TC-OLMA.key', 'overlapping_mentions/TC-OLMA-6.response')
  assert evaluate(doc, muc) == approx([1/2, 2/3, 4/7], abs=TOL)
  assert evaluate(doc, b_cubed) == (11/24, 7/10, 77/139)
  # assert evaluate(doc, mention_overlap) == (9/23, 1, 2*9/23/(1+9/23))

def test_OLMAA1():
  doc = read('overlapping_mentions/TC-OLMAA.key', 'overlapping_mentions/TC-OLMAA-1.response')
  assert evaluate(doc, mention_overlap) == (1, 1, 1)

def test_OLMAA2():
  doc = read('overlapping_mentions/TC-OLMAA.key', 'overlapping_mentions/TC-OLMAA-2.response')
  assert evaluate(doc, mention_overlap) == approx([9/17, 9/10, 2*9/17*9/10/(9/10 + 9/17)], abs=TOL)

def test_OLMB1():
  doc = read('overlapping_mentions/TC-OLMB.key', 'overlapping_mentions/TC-OLMB-1.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, mention_overlap) == (1, 1, 1)

def test_OLMB2():
  doc = read('overlapping_mentions/TC-OLMB.key', 'overlapping_mentions/TC-OLMB-2.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, mention_overlap) == (9/23, 1, 2*9/23/(1+9/23))

def test_OLMB3():
  doc = read('overlapping_mentions/TC-OLMB.key', 'overlapping_mentions/TC-OLMB-3.response')
  #new alighment can find perfect match
  assert evaluate(doc, muc) == (1,1,1)#(3/4, 3/4, 3/4)
  assert evaluate(doc, b_cubed) == (1,1,1)#approx([3/4, 5/6, 15/19], abs=TOL)
  assert evaluate(doc, mention_overlap) == (13/23, 1, 2*13/23/(1+13/23))

def test_OLMB4():
  doc = read('overlapping_mentions/TC-OLMB.key', 'overlapping_mentions/TC-OLMB-4.response')
  assert evaluate(doc, muc) == (3/4, 3/4, 3/4)
  assert evaluate(doc, b_cubed) == (17/24, 17/24, 17/24)
  # assert evaluate(doc, mention_overlap) == (13/23, 1, 2*13/23/(1+13/23))

def test_OLMB5():
  doc = read('overlapping_mentions/TC-OLMB.key', 'overlapping_mentions/TC-OLMB-5.response')
  assert evaluate(doc, muc) == (3/4, 3/3, 6/7)
  assert evaluate(doc, b_cubed) == (17/24, 5/5, 34/41)
  assert evaluate(doc, mention_overlap) == (17/23, 1, 2*17/23/(1+17/23))

def test_OLMB6():
  doc = read('overlapping_mentions/TC-OLMB.key', 'overlapping_mentions/TC-OLMB-6.response')
  assert evaluate(doc, muc) == approx([1/2, 2/3, 4/7], abs=TOL)
  assert evaluate(doc, b_cubed) == (11/24, 7/10, 77/139)
  # assert evaluate(doc, mention_overlap) == (9/23, 1, 2*9/23/(1+9/23))

def test_OLMC1():
  doc = read('overlapping_mentions/TC-OLMC.key', 'overlapping_mentions/TC-OLMC-1.response')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)

def test_OLMC2():#the test case seems not correct
  doc = read('overlapping_mentions/TC-OLMC.key', 'overlapping_mentions/TC-OLMC-2.response')
  # assert evaluate(doc, muc) == approx([3/4, 2/3, 12/17], abs=TOL)
  # assert evaluate(doc, b_cubed) == approx([3/4, 4/5, 24/31], abs=TOL)
  # corrected
  assert evaluate(doc, muc) == approx([2 / 4, 2 / 3, 4 / 7], abs=TOL)
  assert evaluate(doc, b_cubed) == approx([3 / 6, 4 / 5, 8 / 13], abs=TOL)

def test_OLMC3():
  doc = read('overlapping_mentions/TC-OLMC.key', 'overlapping_mentions/TC-OLMC-3.response')
  assert evaluate(doc, muc) == approx([3/4, 1, 6/7], abs=TOL)
  assert evaluate(doc, b_cubed) == approx([17/24, 1, 34/41], abs=TOL)

def test_OLMC4():
  doc = read('overlapping_mentions/TC-OLMC.key', 'overlapping_mentions/TC-OLMC-4.response')
  assert evaluate(doc, muc) == approx([1, 1, 1], abs=TOL)
  assert evaluate(doc, b_cubed) == approx([1, 1, 1], abs=TOL)

# def test_OLMC5():
#   doc = read('overlapping_mentions/TC-OLMC.key', 'overlapping_mentions/TC-OLMC-5.response')
#   assert evaluate(doc, muc) == approx([1, 1, 1], abs=TOL)
#   assert evaluate(doc, b_cubed) == approx([1, 1, 1], abs=TOL)
#   assert evaluate(doc, ceafe) == (38/45, 38/45, 38/45)
#   assert evaluate(doc, ceafm) == (1, 3/4, 6/7)
#   assert evaluate(doc, mention_overlap) == (19/23, 19/26, 2*19/23*19/26/(19/26+19/23))
#
# def test_OLMC6():
#   doc = read('overlapping_mentions/TC-OLMC.key', 'overlapping_mentions/TC-OLMC-6.response')
#   assert evaluate(doc, muc) == approx([1, 2/3, 4/5], abs=TOL)
#   assert evaluate(doc, b_cubed) == approx([1, 71/120, 142/191], abs=TOL)
#   assert evaluate(doc, ceafe) == (38/45, 38/45, 38/45)
#   assert evaluate(doc, ceafm) == (1, 3/4, 6/7)
#   assert evaluate(doc, mention_overlap) == (19/23, 19/28, 2*19/23*19/28/(19/28+19/23))
#
# def test_OLMC7():
#   doc = read('overlapping_mentions/TC-OLMC.key', 'overlapping_mentions/TC-OLMC-7.response')
#   assert evaluate(doc, muc) == approx([1, 3/4, 6/7], abs=TOL)
#   assert evaluate(doc, b_cubed) == approx([1, 79/120, 158/199], abs=TOL)
#   assert evaluate(doc, ceafe) == approx([11/15, 11/15, 11/15], abs=TOL)
#   assert evaluate(doc, ceafm) == approx([1, 3/5, 3/4], abs=TOL)
#   assert evaluate(doc, mention_overlap) == (20/23, 20/40, 2*20/23*20/40/(20/40+20/23))

