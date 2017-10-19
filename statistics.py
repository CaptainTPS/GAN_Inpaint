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
    return sqrt(sum(power(vecA.feature - vecB.feature, 2)))

# generate centers
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def runkMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment

def showResult():
    pass

def kmeans():
    id = 1
    filename = "feature"+str(id)+".txt"
    file = open(filename)
    buffer = file.readlines()
    buffer = [x.split(' ') for x in buffer]
    buffer = [[float(x) for x in line if x != '\n'] for line in buffer]
    items = []
    for i in buffer:
        items.append(feature_item(id, i))

    #running kmeans algorithm


    # print(data)
    pass

#here is the running part
if __name__ == "__main__":
    kmeans()
    pass