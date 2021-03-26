from pytest import approx
from coval.ua.reader import get_coref_infos
from coval.eval.evaluator import evaluate_documents as evaluate
from coval.eval.evaluator import muc, b_cubed, ceafe, lea, ceafm,blancc,blancn

TOL = 1e-4
#the test for blanc is not yet finished

def read(key, response):
  doc_coref_infos, _, _ = get_coref_infos('plural-tests/%s' % key, 'plural-tests/%s' % response,
      True, True, False,False,False,False)
  return doc_coref_infos

def test_PA1():
  doc = read('TC-PA.key', 'TC-PA-1.sys')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, ceafe) == (1, 1, 1)
  assert evaluate(doc, ceafm) == (1, 1, 1)
  assert evaluate(doc, lea) == (1, 1, 1)
  assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA2():
  doc = read('TC-PA.key', 'TC-PA-2.sys')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, ceafe) == (1, 1, 1)
  assert evaluate(doc, ceafm) == (1, 1, 1)
  assert evaluate(doc, lea) == (1, 1, 1)
  assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA3():
  doc = read('TC-PA.key', 'TC-PA-3.sys')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, ceafe) == (1, 1, 1)
  assert evaluate(doc, ceafm) == (1, 1, 1)
  assert evaluate(doc, lea) == (1, 1, 1)
  assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA4():
  doc = read('TC-PA.key', 'TC-PA-4.sys')
  assert evaluate(doc, muc) == approx([3/4, 1, 6/7])
  assert evaluate(doc, b_cubed) == approx([19/24, 1, 38/43])
  assert evaluate(doc, ceafe) == approx([3.8/4, 3.8/4, 3.8/4],abs=TOL)
  assert evaluate(doc, ceafm) == approx([7/8, 1, 0.93333],abs=TOL)
  assert evaluate(doc, lea) == approx([6/8, 1, 6/7])
  #assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA5():
  doc = read('TC-PA.key', 'TC-PA-5.sys')
  assert evaluate(doc, muc) == approx([1, 11/12, 22/23])
  assert evaluate(doc, b_cubed) == approx([1, 0.95167, 0.97523],abs=TOL)
  assert evaluate(doc, ceafe) == approx([0.98333, 0.98333, 0.98333],abs=TOL)
  assert evaluate(doc, ceafm) == approx([0.98611, 0.98611, 0.98611],abs=TOL)
  assert evaluate(doc, lea) == approx([1, 7.6/8, 38/39])
  #assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA6():
  doc = read('TC-PA.key', 'TC-PA-6.sys')
  assert evaluate(doc, muc) == approx([7/8, 7/8, 7/8])
  assert evaluate(doc, b_cubed) == approx([0.88542, 0.92130, 0.903],abs=TOL)
  assert evaluate(doc, ceafe) == approx([0.95833, 0.95833, 0.95833], abs=TOL)
  assert evaluate(doc, ceafm) == approx([0.94643, 0.94643, 0.94643], abs=TOL)
  assert evaluate(doc, lea) == approx([7/8, 11/12, 0.89535], abs=TOL)
  #assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA7():
  doc = read('TC-PA.key', 'TC-PA-7.sys')
  assert evaluate(doc, muc) == approx([5/8, 5/6, 5/7])
  assert evaluate(doc, b_cubed) == approx([0.72461, 1, 0.84032],abs=TOL)
  assert evaluate(doc, ceafe) == approx([0.90278, 0.90278, 0.90278],abs=TOL)
  assert evaluate(doc, ceafm) == approx([0.85714, 0.97959, 0.91429],abs=TOL)
  assert evaluate(doc, lea) == approx([5/8, 16/21, 0.68670], abs=TOL)
  #assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA8():
  doc = read('TC-PA.key', 'TC-PA-8.sys')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, ceafe) == (1, 1, 1)
  assert evaluate(doc, ceafm) == (1, 1, 1)
  assert evaluate(doc, lea) == (1, 1, 1)
  assert evaluate(doc, [blancc, blancn]) == (1, 1, 1)

def test_PA9():
  doc = read('TC-PA.key', 'TC-PA-9.sys')
  assert evaluate(doc, muc) == approx([1, 11/15, 11/13])
  assert evaluate(doc, b_cubed) == approx([1, 0.75463, 0.86016],abs=TOL)
  assert evaluate(doc, ceafe) == approx([0.69167, 0.92222, 0.79048],abs=TOL)
  assert evaluate(doc, ceafm) == approx([0.86111, 0.86111, 0.86111],abs=TOL)
  assert evaluate(doc, lea) == approx([7/8, 5.2/8, 0.74590], abs=TOL)
  #assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA10():
  doc = read('TC-PA.key', 'TC-PA-10.sys')
  assert evaluate(doc, muc) == approx([3/4, 3/4, 3/4])
  assert evaluate(doc, b_cubed) == approx([0.83333, 0.83333, 0.83333],abs=TOL)
  assert evaluate(doc, ceafe) == approx([3.8/4, 3.8/5, 0.84444],abs=TOL)
  assert evaluate(doc, ceafm) == approx([7/8, 7/9, 0.82353],abs=TOL)
  assert evaluate(doc, lea) == approx([6/8, 7/9, 0.76364], abs=TOL)
  #assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PA11():
  doc = read('TC-PA.key', 'TC-PA-11.sys')
  assert evaluate(doc, muc) == approx([7/8, 1, 14/15])
  assert evaluate(doc, b_cubed) == approx([0.88542, 1, 0.93923],abs=TOL)
  assert evaluate(doc, ceafe) == approx([0.97222, 0.97222, 0.97222],abs=TOL)
  assert evaluate(doc, ceafm) == approx([0.95833, 0.95833, 0.95833],abs=TOL)
  assert evaluate(doc, lea) == approx([7/8, 1, 14/15])
  #assert evaluate(doc, [blancc,blancn]) == (1,1,1)


def test_PB1():
  doc = read('TC-PB.key', 'TC-PB-1.sys')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, ceafe) == (1, 1, 1)
  assert evaluate(doc, ceafm) == (1, 1, 1)
  assert evaluate(doc, lea) == (1, 1, 1)
  assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PB2():
  doc = read('TC-PB.key', 'TC-PB-2.sys')
  assert evaluate(doc, muc) == (1, 1, 1)
  assert evaluate(doc, b_cubed) == (1, 1, 1)
  assert evaluate(doc, ceafe) == (1, 1, 1)
  assert evaluate(doc, ceafm) == (1, 1, 1)
  assert evaluate(doc, lea) == (1, 1, 1)
  assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PB3():
  doc = read('TC-PB.key', 'TC-PB-3.sys')
  assert evaluate(doc, muc) == approx([1, 14/15, 28/29])
  assert evaluate(doc, b_cubed) == approx([1, 0.928, 0.96266],abs=TOL)
  assert evaluate(doc, ceafe) == approx([4.9/5, 4.9/5, 4.9/5])
  assert evaluate(doc, ceafm) == approx([0.975, 0.975, 0.975])
  assert evaluate(doc, lea) == approx([1, 0.92, 0.95833],abs=TOL)
  # assert evaluate(doc, [blancc,blancn]) == (1,1,1)

def test_PB4():
  doc = read('TC-PB.key', 'TC-PB-4.sys')
  assert evaluate(doc, muc) == approx([0.8, 0.8, 0.8])
  assert evaluate(doc, b_cubed) == approx([0.85, 7.5/9, 0.84158],abs=TOL)
  assert evaluate(doc, ceafe) == approx([0.77143, 0.96429, 0.85714],abs=TOL)
  assert evaluate(doc, ceafm) == approx([0.8, 8/9, 0.84211],abs=TOL)
  assert evaluate(doc, lea) == approx([0.8, 7/9, 0.78873],abs=TOL)
  # assert evaluate(doc, [blancc,blancn]) == (1,1,1)