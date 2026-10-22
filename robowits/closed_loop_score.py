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


def choose(scores, margin=.02):
    # Conservative fallback: compare only if both candidates are observable.
    if len(scores) != 2:
        raise ValueError('Pilot requires exactly two candidates')
    if all(s['valid'] for s in scores) and scores[1]['score'] > scores[0]['score'] + margin:
        return 1
    return 0
