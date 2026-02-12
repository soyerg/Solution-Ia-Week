"""
VLLM Defect Analyzer — Uses Ollama (LLaVA) to describe casting defects.

Only called when the SVM classifies a piece as DEFECTIVE.
Returns a structured diagnosis: category + short professional description.
"""

import base64
import logging
import re

import httpx

logger = logging.getLogger("vllm_analyzer")

OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "llava:13b"
OLLAMA_TIMEOUT = 60.0  # seconds

# Allowed defect categories (French)
VALID_CATEGORIES = {"trou", "bavure", "déformation", "fissure"}

SYSTEM_PROMPT = (
    "Tu es un inspecteur qualité industriel expert en pièces de fonderie (casting). "
    "L'image montre une pièce de fonderie métallique circulaire (type impeller/roue). "
    "IMPORTANT : cette pièce possède NORMALEMENT un trou central et des formes creuses au milieu — "
    "ce sont des caractéristiques normales de la pièce, PAS des défauts.\n\n"
    "Un système de contrôle automatique a détecté un DÉFAUT sur cette pièce. "
    "Les défauts se trouvent sur les BORDS, la SURFACE ou le CONTOUR extérieur de la pièce. "
    "Cherche des anomalies sur le pourtour et la surface : "
    "fissures (lignes/craquelures), bavures (excès de matière sur les bords), "
    "trous anormaux (petits trous non prévus dans la surface), "
    "déformations (zones enfoncées, bosses, irrégularités de forme).\n\n"
    "Réponds UNIQUEMENT au format suivant, sans rien d'autre :\n"
    "CATÉGORIE — Description courte\n\n"
    "Règles strictes :\n"
    "- La CATÉGORIE doit être exactement l'une de : Trou, Bavure, Déformation, Fissure\n"
    "- La description est UNE SEULE phrase professionnelle de 15 mots maximum\n"
    "- Décris la localisation du défaut sur le BORD ou la SURFACE (pas le centre)\n"
    "- Pas d'introduction, pas de conclusion, pas de texte supplémentaire\n\n"
    "Exemples de réponses valides :\n"
    "Fissure — Craquelure visible sur le bord supérieur du contour extérieur\n"
    "Bavure — Excès de matière le long du bord droit de la pièce\n"
    "Déformation — Enfoncement irrégulier sur la surface supérieure gauche"
)


async def analyze_defect(image_bytes: bytes) -> dict | None:
    """
    Analyze a defective casting image using Ollama LLaVA.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG)

    Returns:
        {"category": "Bavure", "description": "..."} or None on failure
    """
    try:
        # Encode image to base64
        img_b64 = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": SYSTEM_PROMPT,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,      # Low temperature for consistent output
                "num_predict": 80,       # Short response
                "top_p": 0.9,
            },
        }

        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )

        if response.status_code != 200:
            logger.warning(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()
        raw_text = data.get("response", "").strip()
        logger.info(f"Ollama raw response: {raw_text!r}")

        return _parse_diagnosis(raw_text)

    except httpx.ConnectError:
        logger.warning("Ollama service unavailable (connection refused)")
        return None
    except httpx.TimeoutException:
        logger.warning(f"Ollama timed out after {OLLAMA_TIMEOUT}s")
        return None
    except Exception as e:
        logger.error(f"VLLM analysis failed: {e}", exc_info=True)
        return None


async def check_ollama_ready() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


def _parse_diagnosis(text: str) -> dict | None:
    """
    Parse LLaVA response into structured diagnosis.
    Expected format: "Catégorie — Description"
    """
    if not text:
        return None

    # Try to match "CATÉGORIE — description" or "CATÉGORIE - description"
    match = re.match(
        r"^\s*(Trou|Bavure|Déformation|Fissure|Deformation)\s*[—–\-]\s*(.+)",
        text,
        re.IGNORECASE,
    )

    if match:
        category = match.group(1).strip().capitalize()
        description = match.group(2).strip().rstrip(".")
        # Normalize "Deformation" → "Déformation"
        if category.lower() == "deformation":
            category = "Déformation"
        return {"category": category, "description": description}

    # Fallback: try to find any category keyword in the text
    text_lower = text.lower()
    for cat in ["fissure", "bavure", "trou", "déformation", "deformation"]:
        if cat in text_lower:
            category = cat.capitalize()
            if category == "Deformation":
                category = "Déformation"
            # Use the full text as description, cleaned up
            desc = text.split("\n")[0].strip()
            if len(desc) > 100:
                desc = desc[:97] + "..."
            return {"category": category, "description": desc}

    # Last resort: return raw text as description with unknown category
    desc = text.split("\n")[0].strip()
    if len(desc) > 100:
        desc = desc[:97] + "..."
    return {"category": "Défaut", "description": desc}
