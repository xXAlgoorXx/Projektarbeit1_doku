import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import torch
import clip
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

outputfolder = Path("images")
figname = "3DEmbedding"


# Set device and load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Words to visualize
words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'fruit', 'car', 'bus', 'vehicle', 'dog', 'cat', 'animal']
text = clip.tokenize(words).to(device)

# Get CLIP text embeddings
with torch.no_grad():
    text_features = model.encode_text(text).cpu().numpy()

# Use PCA for dimensionality reduction to 3D
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(text_features)

# Update plot settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 12
})

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D embeddings
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], color='blue')

# Annotate each point with the corresponding word
for i, word in enumerate(words):
    ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], word, fontsize=12)

# Set labels
ax.set_title('3D Visualization of Text Embeddings (CLIP + PCA)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.savefig(outputfolder / figname,dpi = 600, bbox_inches='tight')
# Show the plot
plt.show()
