from Core.provider.rerank import TextRerankerProvider


reranker = TextRerankerProvider(
    model_name="Qwen/Qwen3-Reranker-0.6B",
    device="auto",
    max_length=1024,
    backend="local",
)

query = "How do plants use sunlight?"

documents = [
    "Plants use sunlight during photosynthesis to produce chemical energy.",
    "The Eiffel Tower is located in Paris, France.",
    "Photosynthesis allows plants to convert light energy into food."
]

scores = reranker.rerank(
    query=query,
    documents=documents,
    batch_size=1,
)

for i, score in enumerate(scores):
    print(f"Document {i+1}: {score}")

print("Reranker test successful")