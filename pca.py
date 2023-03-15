from sklearn.decomposition import PCA
import numpy as np

# Creazione di un dataset di esempio
X = np.random.rand(100, 10)

# Applicazione della PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualizzazione dei risultati
print('Varianza spiegata: ', pca.explained_variance_ratio_)
print('Primi 5 componenti principali: ', pca.components_[:5])
print('Dati trasformati: ', X_pca[:5])
