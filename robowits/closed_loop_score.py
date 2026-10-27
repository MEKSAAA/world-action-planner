"""Fixed image-only pilot score; deliberately not a learned success detector."""
import cv2
import numpy as np


def components(image, low, high, min_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [(int(stats[i, cv2.CC_STAT_AREA]), centers[i]) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def visual_score(prediction):
    """Score last five ego frames by visible lemon-to-target distance.

    Missing lemon/target or multiple yellow objects is invalid, not success.
    All thresholds are fixed before closed-loop evaluation. No simulator
    state, future observations, rewards or object masks are accepted here.
    """
    distances = []
    for image in prediction[-5:, :224, :224]:
        yellow = components(image, (12, 80, 35), (40, 255, 255), 4)
        green = components(image, (40, 80, 35), (90, 255, 255), 12)
        if len(yellow) != 1 or not green:
            continue
        target = max(green, key=lambda x: x[0])[1]
        distances.append(float(np.linalg.norm(yellow[0][1]-target)/224))
    valid = len(distances) >= 3
    return {'valid': valid, 'valid_frames': len(distances),
            'score': -float(np.mean(distances)) if valid else None,
            'meaning': 'negative 2D lemon-target distance; not task success'}


def choose(scores, margin=.02, require_baseline=True):
    """Rank any number of valid predictions; ties favor the first candidate.

    Legacy callers require a valid candidate 0 for conservative comparisons.
    The action loop instead picks the best valid candidate, even if 0 is invalid.
    If every prediction is invalid, fall back to candidate 0.
    """
    if not scores or not np.isfinite(margin) or margin < 0:
        raise ValueError('Require nonempty scores and a finite nonnegative margin')
    valid = [i for i, s in enumerate(scores) if s.get('valid')
             and s.get('score') is not None and np.isfinite(s['score'])]
    if not valid or (require_baseline and 0 not in valid):
        return 0
    best = max(valid, key=lambda i: scores[i]['score'])
    if 0 in valid and scores[best]['score'] <= scores[0]['score'] + margin:
        return 0
    return best
