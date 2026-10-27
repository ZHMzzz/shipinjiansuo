"""Bounded screenshot refinement with geometrically verified crop alignment."""
import cv2
import numpy as np


def descriptor(frame):
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32).reshape(-1)
    gray -= gray.mean()
    return gray / max(float(np.linalg.norm(gray)), 1e-8)


class ScreenshotMatcher:
    def __init__(self, image):
        self.target = descriptor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        scale = min(1, 640 / max(gray.shape))
        self.gray = cv2.resize(gray, (round(gray.shape[1]*scale), round(gray.shape[0]*scale)))
        self.sift = cv2.SIFT_create(nfeatures=1200)
        self.keypoints, self.features = self.sift.detectAndCompute(self.gray, None)

    def aligned_score(self, frame):
        if self.features is None or len(self.features) < 8: return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1, 640 / max(gray.shape))
        gray = cv2.resize(gray, (round(gray.shape[1]*scale), round(gray.shape[0]*scale)))
        keypoints, features = self.sift.detectAndCompute(gray, None)
        if features is None or len(features) < 2: return None
        pairs = cv2.BFMatcher().knnMatch(self.features, features, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .7*b.distance]
        if len(good) < 8: return None
        source = np.float32([self.keypoints[m.queryIdx].pt for m in good])
        target = np.float32([keypoints[m.trainIdx].pt for m in good])
        transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 3)
        if transform is None or not np.isfinite(transform).all(): return None
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < 8 or inliers.mean() < .35: return None
        height, width = self.gray.shape
        coverage = cv2.contourArea(cv2.convexHull(source[inliers]))/(height*width)
        if coverage < .04: return None
        corners = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]).reshape(1,4,2)
        projected = cv2.perspectiveTransform(corners, transform)[0]
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected): return None
        fh, fw = gray.shape
        area = cv2.contourArea(projected)/(fh*fw)
        if not .01 <= area <= 1.1: return None
        if np.any(projected < [-.05*fw,-.05*fh]) or np.any(projected > [1.05*fw,1.05*fh]): return None
        try: inverse = np.linalg.inv(transform)
        except np.linalg.LinAlgError: return None
        aligned = cv2.warpPerspective(gray, inverse, (width,height))
        # Avoid constant-image correlation degeneracy; features alone are not a match.
        if self.gray.std() < 1 or aligned.std() < 1: return None
        score = float(cv2.matchTemplate(aligned, self.gray, cv2.TM_CCOEFF_NORMED)[0,0])
        return score if np.isfinite(score) else None

    def refine(self, path, max_seconds=60):
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0: return None
            stride = max(1,round(fps/10))
            best = (-2.,0.,None,'whole_frame')
            coarse = []
            index = 0
            while index/fps < max_seconds:
                ok,frame = capture.read()
                if not ok: break
                if index % stride == 0:
                    score = float(self.target @ descriptor(frame))
                    if score > best[0]: best=(score,index/fps,frame.copy(),'whole_frame')
                if index % max(1,round(fps)) == 0: coarse.append(index)
                index += 1
            if best[2] is None: return None
            # Full-frame screenshots retain the existing fast 10Hz matcher.
            if best[0] >= .85: return best
            local = (-2.,0.,None,'aligned_crop')
            for number in coarse:
                capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                ok,frame = capture.read()
                if not ok: continue
                score = self.aligned_score(frame)
                if score is not None and score > local[0]: local=(score,number/fps,frame.copy(),'aligned_crop')
            if local[2] is not None:
                center = round(local[1]*fps)
                for number in range(max(0,center-round(fps)),min(index,center+round(fps)+1),stride):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                    ok,frame = capture.read()
                    if not ok: continue
                    score = self.aligned_score(frame)
                    if score is not None and score > local[0]: local=(score,number/fps,frame.copy(),'aligned_crop')
                if local[0] > best[0]: best=local
            return best
        finally:
            capture.release()
