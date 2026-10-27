import cv2
import numpy as np
from PIL import Image
from scripts.screenshot_search import ScreenshotMatcher, descriptor


def scene(seed):
    rng=np.random.default_rng(seed)
    image=np.zeros((640,360,3),dtype=np.uint8)
    for _ in range(160):
        p=tuple(int(x) for x in rng.integers([0,0],[360,640]))
        cv2.circle(image,p,int(rng.integers(3,18)),tuple(int(x) for x in rng.integers(20,255,3)),-1)
    return image


def test_scaled_crop_alignment_does_not_confuse_unrelated_video():
    frame=scene(7);crop=cv2.resize(frame[80:380,10:350],(680,600))
    matcher=ScreenshotMatcher(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
    assert float(matcher.target@descriptor(frame)) < .5
    assert matcher.aligned_score(frame) > .9
    assert matcher.aligned_score(scene(21)) is None


def test_featureless_image_has_no_geometric_match():
    matcher=ScreenshotMatcher(Image.new('RGB',(100,100),'white'))
    assert matcher.aligned_score(scene(4)) is None
