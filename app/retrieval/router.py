"""Deterministic supervisor routing for the existing retrieval workflow."""


def route_query(query: str, requested_route: str | None = None) -> str:
    """Choose a retrieval modality while preserving explicit caller choices."""
    if requested_route and requested_route.strip().lower() in {"text", "image", "multimodal"}:
        return requested_route.strip().lower()

    query_lower = query.lower()
    if any(word in query_lower for word in ("both", "all", "multimodal", "mix", "hybrid")):
        return "multimodal"
    if any(word in query_lower for word in ("image", "photo", "picture", "chart", "diagram")):
        return "image"
    return "text"