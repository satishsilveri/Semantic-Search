from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed, effective_n_jobs
import numpy as np
import psutil

# Function to get available memory in gigabytes
def get_available_memory():
    return psutil.virtual_memory().available / (1024 ** 3)

# Generate 100,000 random embeddings for demonstration purposes
num_embeddings = 100000
embedding_size = 1024  # Update the embedding size
embeddings = np.random.rand(num_embeddings, embedding_size)

# Create a memory-mapped array for the embeddings
embeddings_filename = "embeddings.dat"
embeddings_memmap = np.memmap(embeddings_filename, dtype='float32', mode='w+', shape=embeddings.shape)
embeddings_memmap[:] = embeddings[:]

# Function to compute pairwise distances for a subset of the data
def compute_distances(start, end):
    subset_embeddings = embeddings_memmap[start:end, :]
    distances = pairwise_distances(subset_embeddings, embeddings_memmap, n_jobs=1)  # n_jobs=1 to avoid nested parallelism
    return distances

# Determine the number of jobs based on available memory
available_memory_gb = get_available_memory()
max_jobs = psutil.cpu_count(logical=False)  # Use the number of physical cores
n_jobs = min(int(available_memory_gb / 4), max_jobs)  # Use 4GB per job as a heuristic

# Split the computation into chunks based on the number of parallel jobs
chunk_size = num_embeddings // n_jobs
chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_jobs - 1)]
chunks.append(((n_jobs - 1) * chunk_size, num_embeddings))

# Use Parallel and delayed from joblib to parallelize the computation
result = Parallel(n_jobs=n_jobs, backend="threading")(delayed(compute_distances)(*chunk) for chunk in chunks)

# Combine the results from each chunk
pairwise_distances_result = np.vstack(result)

# Clean up the memory-mapped array
del embeddings_memmap
