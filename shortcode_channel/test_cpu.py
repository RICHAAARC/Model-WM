"""Offline checks only. The synthetic images and logits carry no scientific evidence."""
import json
import unittest
import numpy as np
from probe_selection.rules import signs
from . import design as D
from . import freeze as F
from . import interface as I

A=json.loads((F.HERE/'allocation.json').read_text())
DEV=[w for w in A['works'] if w['split']=='DEV']


class Dummy:
    def generate(self,w):
        if w['split']!='DEV': raise AssertionError('Forbidden stage')
        clean=np.full((32,32,3),127,dtype=np.uint8)
        return None,clean,clean

    def write(self,h,code):
        pic=h.copy()
        pic[:16,:16,:]=np.array(list(map(int,code))).reshape(4,4,3).repeat(4,0).repeat(4,1)*50+100
        return pic

    def read(self,rgb):
        return rgb[:16,:16,:].reshape(4,4,4,4,3).mean(axis=(1,3)).reshape(-1)-127


class Checks(unittest.TestCase):
    def test_allocation(self):
        p,s,h=F.history()
        self.assertEqual(h,A['history'])
        self.assertEqual(len(A['works']),48)
        self.assertEqual(len({w['seed'] for w in A['works']}),48)
        self.assertFalse(s & {w['seed'] for w in A['works']})
        self.assertFalse(p & {F.normal(w['prompt']) for w in A['works']})
        self.assertEqual([sum(w['split']==k for w in A['works']) for k in ('DEV','CAL','TEST')],[8,24,16])
        self.assertEqual(len({tuple(g) for w in A['works'] for g in w['groups'][:2]}),96)
        self.assertEqual(len({tuple(w['groups'][2]) for w in A['works']}),48)
        for w in A['works']:
            a,b,c=map(set,w['groups'])
            self.assertFalse(a&b)
            self.assertEqual(len(a|b|c),9)
            self.assertTrue(all(i<16 for i in a|b) and all(i>=16 for i in c))
            self.assertEqual(set(w['book_order']),set(I.BOOKS))

    def test_geometry(self):
        for name in D.LAYOUTS:
            bank=signs(A['books'][name])
            self.assertTrue(np.all(bank.sum(1)==0))
            np.testing.assert_array_equal(bank@bank.T,48*np.eye(32))
            expected=np.zeros((560,16))
            np.put_along_axis(expected,D.TRIPLES,1/3,axis=1)
            np.testing.assert_allclose(D.score(bank[D.TRIPLES].mean(1),bank),expected,atol=1e-14)
        self.assertEqual(len({frozenset(A['books'][k][:16]) for k in D.LAYOUTS}),4)
        self.assertEqual(len(set(A['books']['tardos'])),32)

    def test_synthetic_dev(self):
        ledger=I.CPUCounter()
        result=I.acquire_dev(A,Dummy(),ledger)
        self.assertEqual(dict(ledger.done),I.LIMITS)
        self.assertEqual(ledger.done,ledger.started)
        self.assertEqual(len(result),48)
        d=D.dev_decision(result,A)
        self.assertEqual(len(d['paired_order_counts']),8)
        self.assertFalse(d['eligible_for_separate_CAL_authorization'])
        with self.assertRaises(ValueError): D.dev_decision(result[:-1],A)
        with self.assertRaises(ValueError): D.fit([dict(work='TEST-32',book='H0')],{})
        # Real paired-layout folds exclude held work, both layouts and true triples.
        for w in DEV:
            forbidden={tuple(x) for x in w['groups'][:2]}
            train=[r for r in result if r['work']!=w['id'] and r['book'] in ('H2','H3')]
            self.assertFalse(forbidden & {tuple(g) for r in train for g in r['groups'][:2]})

    def test_cal_mock_and_test_boundaries(self):
        row=np.zeros(16);row[1]=5;row[2]=2
        threshold=D.calibrate([[(row,[1])]]*24)
        d=D.adjudicate(row,threshold,[1,2])
        self.assertTrue(d['safe'] and not d['all_safe'])
        self.assertEqual(d['returned'],[1])
        self.assertTrue(D.adjudicate(row,5,[])['reject'])
        self.assertEqual(D.adjudicate(row,2,[])['wrong'],[1])
        with self.assertRaises(ValueError): D.calibrate([[]]*24)
        with self.assertRaises(ValueError): D.adjudicate(np.ones(15),3,[])
        with self.assertRaises(ValueError): D.native_or_tardos([np.nan]*48,signs(A['books']['H0']))
        self.assertEqual(D.paired([True]*5+[False]*11,[False]*16),(5,1/32))
        self.assertEqual(D.paired([True]*4+[False]*12,[False]*16),(4,1/16))
        b=signs(A['books']['tardos'])
        self.assertEqual(D.native_or_tardos(b[0],b,np.array(A['tardos_biases'])).shape,(16,))

    def test_interaction_nested_in_ordinary_model(self):
        rng=np.random.default_rng(98)
        banks={k:signs(v) for k,v in A['books'].items()}
        families=[]
        for w in DEV:
            for book in D.LAYOUTS:
                bank=banks[book]
                r=dict(work=w['id'],book=book,groups=w['groups'],reads={})
                for g in w['groups'][:2]:
                    for i in g:
                        common=rng.normal(0,.1,24).repeat(2)
                        q=bank[i]+common+rng.normal(0,.0001,48)
                        r['reads'][f'u{i}']={'jpeg85':q}
                for n,g in enumerate(w['groups'][:2]):
                    r['reads'][f'm{n}']={'jpeg85':bank[g].mean(0)}
                families.append(r)
        initial=D.fit(families,banks)
        for r in families:
            bank=banks[r['book']]
            for n,g in enumerate(r['groups'][:2]):
                z=bank[g].mean(0)
                r['reads'][f'm{n}']['jpeg85']=z+3*z*(np.abs(z[initial['neighbor']])<.5)
        m=D.fit(families,banks)
        self.assertGreater(m['beta'],2)
        self.assertTrue(np.array_equal(initial['neighbor'],m['neighbor']))
        bank=banks['H0']
        unchanged=dict(m,beta=0.)
        np.testing.assert_allclose(D.predict(unchanged,bank,D.TRIPLES,True),
                                   D.predict(m,bank,D.TRIPLES,False))
        self.assertGreater(np.linalg.norm(D.predict(m,bank,D.TRIPLES,True)-D.predict(m,bank,D.TRIPLES,False)),1)

    def test_blind_slots(self):
        for w in A['works']:
            self.assertEqual(len({I.blind_ticket(w,b) for b in I.BOOKS}),6)
            self.assertTrue(all('H' not in I.blind_ticket(w,b) for b in D.LAYOUTS))

    def test_budget_and_failure(self):
        self.assertEqual(F.budget(),A['budget'])
        ledger=I.CPUCounter()
        with self.assertRaises(ZeroDivisionError):
            ledger('write','x',lambda:1/0)
        self.assertEqual(ledger.started['write'],1)
        self.assertEqual(ledger.done['write'],0)
        ledger.started['write']=432
        with self.assertRaises(RuntimeError):
            ledger('write','x',lambda:True)
        with self.assertRaises(ValueError):
            I.query(lambda x:[1]*47,np.zeros((32,32,3),np.uint8))

if __name__=='__main__':
    unittest.main(verbosity=2)
