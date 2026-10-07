from Core.provider.embedding import TextEmbeddingProvider


embedder = TextEmbeddingProvider(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    backend="local",
    device="auto",
    max_length=4096,
)

texts = [
    "Photosynthesis converts light energy into chemical energy.",
    "Plants use sunlight to produce food.",
    "The Eiffel Tower is located in Paris."
]

embeddings = embedder.embed_texts(texts)

print("Embedding shape:", embeddings.shape)

similarity_1 = embeddings[0] @ embeddings[1]
similarity_2 = embeddings[0] @ embeddings[2]

print("Related similarity:", similarity_1)
print("Unrelated similarity:", similarity_2)
print("Embedding test successful")