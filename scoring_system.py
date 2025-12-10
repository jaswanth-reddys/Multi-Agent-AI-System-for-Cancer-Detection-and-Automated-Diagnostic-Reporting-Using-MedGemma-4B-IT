import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    VERY_HIGH = (0.85, 1.0, "VERY HIGH")
    HIGH = (0.70, 0.85, "HIGH")
    MODERATE = (0.55, 0.70, "MODERATE")
    LOW = (0.40, 0.55, "LOW")
    VERY_LOW = (0.0, 0.40, "VERY LOW")


@dataclass
class DiagnosisScore:
    malignancy_risk: float
    confidence: float
    confidence_level: str
    weighted_score: float
    keyword_matches: Dict[str, int]
    indicators: Dict[str, bool]
    reasoning: str


class JSONExtractor:
    """Extracts structured values from JSON model output."""

    @staticmethod
    def parse_json(text: str):
        try:
            cleaned = text[text.index("{"): text.rindex("}") + 1]
            return json.loads(cleaned)
        except:
            return None


class CustomScoringSystem:
    """New JSON-aware scoring system with fallback to keyword scoring."""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type.lower()

    def score_analysis(self, analysis_text: str) -> DiagnosisScore:

        # Try extracting JSON first
        json_data = JSONExtractor.parse_json(analysis_text)

        if json_data:
            return self._score_from_json(json_data)
        else:
            return self._score_from_text(analysis_text)

    # --------------------------------------------
    # 1️⃣ JSON SCORING (Primary)
    # --------------------------------------------
    def _score_from_json(self, data: dict) -> DiagnosisScore:

        malignancy_prob = float(data.get("malignancy_probability", data.get(
            "likelihood_malignancy", data.get("risk_score", 0.5))))

        # Convert 0–1 probability into risk score
        malignancy_risk = malignancy_prob

        # Simple confidence mapping
        if malignancy_risk > 0.85:
            confidence = 0.90
        elif malignancy_risk > 0.65:
            confidence = 0.75
        elif malignancy_risk > 0.45:
            confidence = 0.60
        else:
            confidence = 0.45

        confidence_level = self._map_confidence(confidence)

        # Weighted score = combined risk + confidence
        weighted_score = (malignancy_risk * 0.7) + (confidence * 0.3)

        # Indicators
        indicators = {
            "mentions_biopsy": "biopsy" in str(data).lower(),
            "mentions_follow_up": "follow" in str(data).lower(),
            "mentions_urgent": "urgent" in str(data).lower(),
            "has_malignant_keywords": malignancy_risk >= 0.6,
            "has_benign_keywords": malignancy_risk <= 0.4,
        }

        reasoning = f"Model-estimated malignancy probability is {malignancy_risk*100:.1f}%. Confidence level is {confidence_level}."

        return DiagnosisScore(
            malignancy_risk=malignancy_risk,
            confidence=confidence,
            confidence_level=confidence_level,
            weighted_score=weighted_score,
            keyword_matches={},
            indicators=indicators,
            reasoning=reasoning
        )

    # --------------------------------------------
    # 2️⃣ FALLBACK TEXT SCORING (Old system)
    # --------------------------------------------
    def _score_from_text(self, text: str):

        # Basic detection
        malignant_hits = len(re.findall(
            r"malignant|cancer|high|spiculated|necrosis", text.lower()))
        benign_hits = len(re.findall(
            r"benign|normal|stable|low", text.lower()))

        total = malignant_hits + benign_hits
        malignancy_risk = (malignant_hits / total) if total > 0 else 0.5

        confidence = 0.5 + (abs(malignancy_risk - 0.5))
        confidence_level = self._map_confidence(confidence)

        weighted_score = (malignancy_risk * 0.6) + (confidence * 0.4)

        indicators = {
            "mentions_biopsy": "biopsy" in text.lower(),
            "mentions_follow_up": "follow-up" in text.lower(),
            "mentions_urgent": "urgent" in text.lower(),
            "has_malignant_keywords": malignant_hits > 0,
            "has_benign_keywords": benign_hits > 0
        }

        reasoning = f"Keywords suggest malignancy risk of {malignancy_risk*100:.1f}%. Confidence is {confidence_level}."

        return DiagnosisScore(
            malignancy_risk=malignancy_risk,
            confidence=confidence,
            confidence_level=confidence_level,
            weighted_score=weighted_score,
            keyword_matches={"malignant": malignant_hits,
                             "benign": benign_hits},
            indicators=indicators,
            reasoning=reasoning
        )

    # --------------------------------------------
    # Helper
    # --------------------------------------------
    def _map_confidence(self, val: float) -> str:
        for level in ConfidenceLevel:
            if level.value[0] <= val < level.value[1]:
                return level.value[2]
        return "VERY HIGH"
