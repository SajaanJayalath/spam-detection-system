from datetime import datetime
from typing import Dict, Any


def build_response(pred_label: str, class_probs: Dict[str, float]) -> Dict[str, Any]:
    confidence = float(max(class_probs.values())) if class_probs else 0.0
    return {
        "prediction": pred_label,
        "confidence": confidence,
        "class_probs": {k: float(v) for k, v in class_probs.items()},
        "timestamp": datetime.now().isoformat(),
    }
