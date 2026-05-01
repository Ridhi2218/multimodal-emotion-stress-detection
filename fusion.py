# ============================================================
#  FUSION MODULE — Combines face + voice into stress result
#  Implements the fusion formula: S = f(F, V, P)
#  from the research paper
# ============================================================


def full_analysis(face_emotions: dict, voice_stress: float,
                  face_weight: float = 0.6, voice_weight: float = 0.4) -> dict:
    """
    Combines face emotion scores and voice energy into a final
    stress assessment.

    Args:
        face_emotions : dict of emotion → score from face module
        voice_stress  : 0.0-1.0 energy score from voice module
        face_weight   : how much weight to give the face modality
        voice_weight  : how much weight to give the voice modality

    Returns:
        dict with keys:
            stress_score, stress_label, stress_color,
            dominant_emotion, recommendation, fused_emotions
    """
    if not face_emotions:
        face_emotions = {
            'angry': 0.05, 'disgust': 0.02, 'fear': 0.03,
            'happy': 0.20, 'sad': 0.10, 'surprise': 0.05, 'neutral': 0.55
        }

    # Stress-related emotions from face
    face_stress = (
        face_emotions.get('angry',   0) +
        face_emotions.get('fear',    0) +
        face_emotions.get('sad',     0) +
        face_emotions.get('disgust', 0)
    )

    # Weighted fusion: S = w_f * F + w_v * V
    stress_score = face_weight * face_stress + voice_weight * voice_stress
    stress_score = float(min(stress_score, 1.0))

    # Fused emotion scores (shift by voice stress impact)
    fused_emotions = {}
    voice_boost = voice_stress * 0.15
    for em, sc in face_emotions.items():
        if em in ('angry', 'fear', 'sad', 'disgust'):
            fused_emotions[em] = min(sc + voice_boost, 1.0)
        else:
            fused_emotions[em] = max(sc - voice_boost * 0.5, 0.0)

    # Normalise fused so they sum to 1
    total = sum(fused_emotions.values())
    if total > 0:
        fused_emotions = {k: v / total for k, v in fused_emotions.items()}

    # Stress classification
    if stress_score > 0.45:
        label = "HIGH STRESS"
        color = "#EF4444"
        recommendation = (
            "Your stress indicators are elevated. "
            "Take a 5-minute break, practice deep breathing (4-7-8 method), "
            "drink water, and step away from the screen."
        )
    elif stress_score > 0.22:
        label = "MODERATE STRESS"
        color = "#F59E0B"
        recommendation = (
            "Mild stress detected. "
            "Light stretching, calming music, or a short walk may help. "
            "Consider a 2-minute mindfulness pause."
        )
    else:
        label = "LOW STRESS"
        color = "#10B981"
        recommendation = (
            "You appear calm and composed. "
            "Excellent psychophysiological state — maintain it with "
            "regular breaks and good sleep habits. 🌟"
        )

    dominant_emotion = max(face_emotions, key=face_emotions.get)

    return {
        "stress_score":      stress_score,
        "stress_label":      label,
        "stress_color":      color,
        "dominant_emotion":  dominant_emotion,
        "recommendation":    recommendation,
        "fused_emotions":    fused_emotions,
    }
