# ============================================================
#  FUSION MODULE — Combines face + voice → Final stress level
#  This implements the core formula from your paper:
#       S = f(F, V, P)
#  where F=Face, V=Voice, P=Physiological signals
# ============================================================


# ── Emotion categories ────────────────────────────────────────
STRESS_EMOTIONS  = ['angry', 'sad', 'fear', 'disgust', 'fearful']
CALM_EMOTIONS    = ['happy', 'neutral', 'calm', 'surprised']


def normalize_scores(emotions: dict) -> dict:
    """
    Ensures all emotion scores add up to 1.0 (normalization).
    This is needed before fusion so both modalities are on the
    same scale.
    """
    total = sum(emotions.values())
    if total == 0:
        return emotions
    return {k: v / total for k, v in emotions.items()}


def map_voice_stress(voice_stress_score: float) -> dict:
    """
    Converts the raw voice stress score (0–1) into an
    emotion-like dictionary so it can be fused with face emotions.
    """
    return {
        'angry':   voice_stress_score * 0.4,
        'sad':     voice_stress_score * 0.3,
        'fearful': voice_stress_score * 0.3,
        'happy':   (1 - voice_stress_score) * 0.6,
        'neutral': (1 - voice_stress_score) * 0.4,
    }


def weighted_fusion(face_emotions: dict,
                    voice_stress: float,
                    face_weight: float = 0.6,
                    voice_weight: float = 0.4) -> dict:
    """
    Implements the weighted fusion formula from your paper:
        S = w1*F + w2*V

    Face gets 60% weight (more reliable visual cue).
    Voice gets 40% weight.

    Both weights must add up to 1.0.
    """
    # Normalize face emotions
    face_norm = normalize_scores(face_emotions)

    # Convert voice score to emotion dict
    voice_emotions = map_voice_stress(voice_stress)
    voice_norm = normalize_scores(voice_emotions)

    # Get all unique emotion keys
    all_emotions = set(face_norm.keys()) | set(voice_norm.keys())

    # Weighted combination
    fused = {}
    for emotion in all_emotions:
        f_score = face_norm.get(emotion, 0)
        v_score = voice_norm.get(emotion, 0)
        fused[emotion] = (face_weight * f_score) + (voice_weight * v_score)

    return fused


def calculate_stress_level(fused_emotions: dict) -> tuple:
    """
    Maps fused emotions to a final stress level.
    
    Returns:
        stress_label : "HIGH STRESS" / "MODERATE STRESS" / "LOW STRESS"
        stress_score : float between 0 and 1
        color        : color code for UI display
    """
    # Sum up all stress-related emotion scores
    stress_score = sum(
        fused_emotions.get(e, 0) for e in STRESS_EMOTIONS
    )

    # Sum up all calm emotion scores
    calm_score = sum(
        fused_emotions.get(e, 0) for e in CALM_EMOTIONS
    )

    # Normalize stress score to 0–1 range
    total = stress_score + calm_score
    if total > 0:
        normalized_stress = stress_score / total
    else:
        normalized_stress = 0.5

    # Classify into 3 levels
    if normalized_stress >= 0.55:
        return "HIGH STRESS", normalized_stress, "#FF4B4B"
    elif normalized_stress >= 0.30:
        return "MODERATE STRESS", normalized_stress, "#FFA500"
    else:
        return "LOW STRESS", normalized_stress, "#00C851"


def get_recommendation(stress_label: str, dominant_emotion: str) -> str:
    """
    Returns a helpful recommendation based on detected stress.
    """
    recommendations = {
        "HIGH STRESS": [
            "🧘 Try deep breathing: inhale 4 sec, hold 4 sec, exhale 6 sec.",
            "🚶 Take a short walk away from your screen.",
            "💧 Drink some water and take a 10-minute break.",
            "🎵 Listen to calming music to reset your focus.",
        ],
        "MODERATE STRESS": [
            "😌 You seem a little tense. Try rolling your shoulders.",
            "☕ Consider a short break with a warm drink.",
            "📝 Write down what's on your mind to clear your head.",
        ],
        "LOW STRESS": [
            "✅ You're doing great! Keep up the good energy.",
            "😊 Your stress levels look healthy. Stay positive!",
        ]
    }

    import random
    options = recommendations.get(stress_label, ["Stay mindful!"])
    return random.choice(options)


def full_analysis(face_emotions: dict, voice_stress: float) -> dict:
    """
    Main function — runs complete analysis pipeline.
    
    Input:
        face_emotions : dict from face_module.detect_face_emotions()
        voice_stress  : float from voice_module.analyze_voice_energy()
    
    Output:
        Complete analysis result as a dictionary
    """
    # Step 1: Fuse both modalities
    fused = weighted_fusion(face_emotions, voice_stress)

    # Step 2: Calculate stress level
    stress_label, stress_score, color = calculate_stress_level(fused)

    # Step 3: Get dominant emotion
    dominant_emotion = max(face_emotions, key=face_emotions.get)

    # Step 4: Get recommendation
    recommendation = get_recommendation(stress_label, dominant_emotion)

    return {
        "stress_label":    stress_label,
        "stress_score":    round(stress_score, 3),
        "stress_color":    color,
        "dominant_emotion": dominant_emotion,
        "face_emotions":   face_emotions,
        "fused_emotions":  fused,
        "voice_stress":    voice_stress,
        "recommendation":  recommendation,
    }


# Run directly to test this module alone
if __name__ == "__main__":
    print("Testing Fusion Module with sample data...")

    # Simulated face emotions (as if CNN detected these)
    sample_face = {
        'angry':   0.45,
        'sad':     0.20,
        'happy':   0.10,
        'neutral': 0.15,
        'fear':    0.05,
        'disgust': 0.03,
        'surprise':0.02,
    }

    # Simulated voice stress score
    sample_voice_stress = 0.65

    result = full_analysis(sample_face, sample_voice_stress)

    print("\n── FUSION RESULT ──────────────────────────")
    print(f"  Dominant Emotion : {result['dominant_emotion']}")
    print(f"  Stress Level     : {result['stress_label']}")
    print(f"  Stress Score     : {result['stress_score']:.0%}")
    print(f"  Recommendation   : {result['recommendation']}")
    print("───────────────────────────────────────────")
