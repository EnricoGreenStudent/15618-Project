import random

def generateEdgeList(numVertices, numEdges, maxEdgeWeight, testName):
    fileName = testName + ".txt"
    f = open(fileName, "w")
    f.write(str(numVertices) + "\n")
    for i in range(numEdges):
        source = random.randrange(numVertices)
        dest = random.randrange(numVertices)
        weight = random.randrange(maxEdgeWeight) + 1
        edgeRepresentation = str(source) + "," + str(dest) + "," + str(weight) + "\n"
        f.write(edgeRepresentation)
    f.close()

def generateCompleteGraph(numVertices, maxEdgeWeight, testName):
    fileName = testName + ".txt"
    f = open(fileName, "w")
    f.write(str(numVertices) + "\n")
    for i in range(numVertices):
        for j in range(numVertices):
            if i != j:
                edgeRepresentation = str(i) + "," + str(j) + "," + str(random.randrange(maxEdgeWeight) + 1) + "\n"
                f.write(edgeRepresentation)
    f.close()

def generateCycle(numVertices, maxEdgeWeight, testName):
    fileName = testName + ".txt"
    f = open(fileName, "w")
    f.write(str(numVertices) + "\n")
    for i in range(numVertices):
        if i == numVertices - 1:
            edgeRepresentation = str(i) + "," + str(0) + "," + str(random.randrange(maxEdgeWeight) + 1) + "\n"
        else:
            edgeRepresentation = str(i) + "," + str(i+1) + "," + str(random.randrange(maxEdgeWeight) + 1) + "\n"
        f.write(edgeRepresentation)
    f.close()

def generateTree(numVertices, maxEdgeWeight, testName):
    fileName = testName + ".txt"
    f = open(fileName, "w")
    f.write(str(numVertices) + "\n")
    for i in range(numVertices - 1):
        weight = random.randrange(maxEdgeWeight) + 1
        otherVertex = random.randrange(i + 1)
        edgeRepresentation = str(i + 1) + "," + str(otherVertex) + "," + str(weight) + "\n"
        f.write(edgeRepresentation)
        edgeRepresentation = str(otherVertex) + "," + str(i+1) + "," + str(weight) + "\n"
        f.write(edgeRepresentation)
    f.close()

# random - 50k vertices, 100k edges
generateEdgeList(50000, 100000, 50, "random")
# complete - 250 vertices
generateCompleteGraph(250, 250, "complete")
# cycle - 100k vertices
generateCycle(100000, 1, "cycle")
# tree - 100k vertices
generateTree(100000, 1, "tree")
# shortTests - quick sanity test, 5 vertices, 10 edges
generateEdgeList(5, 10, 5, "shortTest")

generateEdgeList(1000, 30000, 50, "random-1k")