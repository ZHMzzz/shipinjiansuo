import unittest
from evaluate import condition_filter, best_per_video
from plans import compile_plan

class GateTests(unittest.TestCase):
    def test_same_segment_and_and_alternative_segment(self):
        texts={'hat':{'scores':[.8,.1,.7]},'bike':{'scores':[.1,.8,.7]}}
        self.assertEqual(condition_filter([0,1],[['hat','bike']],texts),[])
        kept=condition_filter([0,1,2],[['hat','bike']],texts)
        self.assertEqual(kept,[2])
        self.assertEqual(best_per_video(kept,[.9,.8,.7],[{'media_id':1}]*3),[2])

    def test_or_preserves_each_branch(self):
        texts={'run':{'scores':[.8,.1]},'fall':{'scores':[.1,.8]}}
        self.assertEqual(condition_filter([0,1],[['run'],['fall']],texts),[0,1])

    def test_empty_group_is_not_match_all(self):
        with self.assertRaises(ValueError):
            condition_filter([0],[[]],{})

    def test_or_subjects_scoped_per_group(self):
        def group(text):return {'must':[text], 'must_not':[], 'prefer':[], 'relations':[]}
        result={'id':'or','query':'OR','schema_valid':True,'parsed':{'groups':[group('人物A跑步'),group('人物B滑倒')],'uncertain':[]}}
        p=compile_plan(result)
        self.assertFalse(p['fallback_reasons'])
        self.assertEqual(p['groups_normalized'],[['有人跑步'],['有人滑倒']])

    def test_negative_fact_in_must_falls_back(self):
        result={'id':'neg','query':'没戴帽','schema_valid':True,'parsed':{'groups':[{'must':['人物A未佩戴安全帽'],'must_not':[],'prefer':[],'relations':[]}],'uncertain':[]}}
        self.assertIn('negative_fact_in_must',compile_plan(result)['fallback_reasons'])

if __name__=='__main__':unittest.main()
