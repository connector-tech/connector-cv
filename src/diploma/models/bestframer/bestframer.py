def bestframe_score(pitch_score: float, yaw_score: float, blur_score: float) -> float:
    score = 0
    if pitch_score > 0.5 or pitch_score < -0.5:
        return score - 10000
    if pitch_score > 0.5 or pitch_score < -0.5:
        return score - 10000
    if blur_score < 25:
        return score - 10000

    if pitch_score > 0.35 or pitch_score < -0.35:
        score -= 300
    elif pitch_score > 0.2 or pitch_score < -0.2:
        score -= 120
    elif pitch_score > 0.1 or pitch_score < -0.1:
        score -= 50

    if yaw_score > 0.35 or yaw_score < -0.35:
        score -= 300
    elif yaw_score > 0.2 or yaw_score < -0.2:
        score -= 120
    elif yaw_score > 0.1 or yaw_score < -0.1:
        score -= 50

    if blur_score < 40:
        score -= 300
    elif blur_score < 70:
        score -= 120
    elif blur_score < 100:
        score -= 50

    score = score - abs(yaw_score) * 100 - abs(pitch_score) * 100 + blur_score * 0.01

    return score
