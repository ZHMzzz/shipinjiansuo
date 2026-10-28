"""Small correctness checks for the offline evaluator, without model inference."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('compound_evaluate',Path(__file__).with_name('evaluate.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_conditions_must_cooccur_in_one_segment_and_an_alternative_can_pass():
    raw={'segments':[{'media_id':mid,'ts':float(ts),'te':float(ts+8)} for mid,ts in [(1,0),(1,8),(2,0),(3,0),(3,8)]],
         'texts':{key:{'scores':scores,'seconds':.1} for key,scores in {
             'query':[.7,.69,.65,.68,.66],
             'A':[.8,.2,.7,.8,.7],
             'B':[.2,.8,.7,.2,.7]}.items()},
         'model_load_seconds':0,'index_before':{},'index_after':{},'pointer_unchanged':True}
    case={'id':'case','split':'calibration','query':'query','conditions':['A','B'],'positive_media':[2,3],'negative_media':[1]}
    design={'selection':'test','rules':{'minimum_condition_scores':[.5],'maximum_condition_gaps':[None]},'cases':[case]}
    report=module.evaluate(design,raw)
    result=report['selected']['cases'][0]
    assert result['after_ids']==[2,3]  # video 1 cannot combine its two windows
    assert result['best_only_ids']==[2]  # video 3 has an alternative valid window
    assert report['selected']['calibration']['positive_deleted']==0
    assert report['selected']['calibration']['false_removed']==1
