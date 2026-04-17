from ..protocols import EmbeddingFactory
from .eres2netv2_embedder import ERes2NetV2EmbeddingFactory


def get_embeddings_factory(device: str) -> EmbeddingFactory:
    return ERes2NetV2EmbeddingFactory(device=device)
