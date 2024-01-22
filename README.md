# Semantic-Search

Repository containing examples of different semantic search pipelines.

## SBERT inference docker commands

Build:
```
docker build --no-cache -t camembert_optimized .
```

Run:
```
docker run --cpus=3 --memory=2g --name camembert_ep -p 8000:8000 -d camembert_optimized
```
