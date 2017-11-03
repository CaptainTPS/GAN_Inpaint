from numpy import *


class feature_item:
    def __init__(self):
        self.id = 0
        self.feature = []

    def __init__(self, id, feature):
        self.id = id
        self.feature = feature


# calculate distance
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# generate centers
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        # temp = dataSet[:, j]
        minJ = min(dataSet[:, j])
        rangeJ = float(max(array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def randPick(dataSet, k):
    n = shape(dataSet)[1]
    ran = shape(dataSet)[0]
    centroids = mat(zeros((k, n)))
    seed = []
    for j in range(k):
        s = random.randint(0, ran - len(seed))
        for m in seed:
            if(s >= m):
                s += 1
        seed.append(s)
        centroids[j] = dataSet[s, :]
    return centroids


def runkMeans(dataSet, k, distMeas=distEclud, createCent=randPick):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True

    maxiter = 10000
    i = 0
    while clusterChanged:

        if i > maxiter:
            break

        if i%1000 == 0:
            print("iter to " + str(i))
            print centroids

        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment

def showResult(centers, clusterAssment, items):
    import matplotlib.pyplot as plt

    k = len(centers)

    for i in range(k):
        arr = []
        # arr = [items[ii].id for ii, x in clusterAssment[:, 0] if int(x.A) == i]

        clen = len(clusterAssment[:, 0])

        for ii in range(clen):
            x = clusterAssment[:, 0][ii]
            if int(x.A) == i:
                arr.append(items[ii].id)

        plt.hist(arr)
        plt.title("Cluster "+str(i)+" Histogram")
        plt.xlabel("Feature id")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig("Cluster"+str(i)+".png")
        plt.close()

    pass

def kmeans():
    items = []
    id = 3
    for idi in range(id):
        filename = "feature"+str(idi)+".data"
        # filename = "/home/cad/PycharmProjects/ContextEncoder/testdata/f" + str(idi) + ".txt"
        file = open(filename)
        buffer = file.readlines()
        buffer = [x.split(' ') for x in buffer]
        buffer = [[float(x) for x in line if x != '\n'] for line in buffer]

        for i in buffer:
            items.append(feature_item(idi, i))
        file.close()

    #running kmeans algorithm
    dataset = mat([x.feature for x in items])
    centers, clusterAssment = runkMeans(dataset, 10)

    showResult(centers, clusterAssment, items)
    # print(data)
    pass

#here is the running part
if __name__ == "__main__":
    kmeans()
    pass