from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt

X, Y = load_digits(return_X_y=True)
print(Y.shape)
print(Y)
print(X.shape)
embedding = SpectralEmbedding(n_components=2)
X_transformed = embedding.fit_transform(X[:100])
print(X_transformed.shape)

fig, ax = plt.subplots()
ax.scatter(X_transformed[:,0], X_transformed[:,1])

for i, txt in enumerate(Y[0:99]):
    ax.annotate(txt, (X_transformed[i,0], X_transformed[i,1]))
#plt.plot(X_transformed[:,0], X_transformed[:,1], 'bo')
#plt.ylabel('some numbers')
plt.show()