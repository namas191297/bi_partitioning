def bi_partitioning(sigma=0.4):
    data = np.load('q3.npy')
    WeightMatrix = np.zeros((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            if i == j:
                WeightMatrix[i][j] = 0
            else:
                WeightMatrix[i][j] = np.exp((-1 * (np.linalg.norm(data[i] - data[j]) ** 2))/(sigma ** 2))
    DegreeMatrix = np.sum(WeightMatrix, axis=1)
    L = np.diag(DegreeMatrix) - WeightMatrix
    DSquareRoot = np.diag(1.0 / (DegreeMatrix ** (0.5)))
    Lnorm = np.dot(np.dot(DSquareRoot, L), DSquareRoot)

    eigvals, eigvecs = np.linalg.eig(Lnorm)
    eigvecs = np.array(eigvecs, dtype=np.float64)
    finaleigvec = eigvecs[:, 1]

    cluster1 = []
    cluster2 = []
    for i in range(len(finaleigvec)):
        if finaleigvec[i] > 0:
            cluster1.append(data[i])
        elif finaleigvec[i] < 0:
            cluster2.append(data[i])
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)

    plt.scatter(cluster1[:, 0], cluster1[:, 1])
    plt.scatter(cluster2[:, 0], cluster2[:, 1])
    plt.title('Bi-partitioned Data')
    plt.show()

bi_partitioning()