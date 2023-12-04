#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "timing.h"
#include "dijkstra.cpp"
#include "bellman_forward.cpp"
#include "bellman_backward.cpp"
#include "delta-step.cpp"

// Prints a list of edges in the graph. Used for testing
void testGraphLoading(graph g) {
    for(int i = 0; i < g.vertices.size(); i++) {
        for(int j = 0; j < g.vertices[i].size(); j++) {
            printf("%d %d %f\n", i, g.vertices[i][j].dest, g.vertices[i][j].weight);
        }
    }
}

void dummyFunc(graph g, std::vector<float> dists) {
    sleep(1);
    for(int i = 0; i < g.numVertices; i++) {
        dists.push_back(i);
    }
}

// Outputs the SSSP results to a file
void saveResults(std::vector<float> dists, std::string fileName) {
    std::ofstream file(fileName);
    for(int i = 0; i < dists.size(); i++) {
        file << std::to_string(i);
        file << ": ";
        file << std::to_string(dists[i]);
        file << "\n";
    }
    file.close();
}

// Compares two files line-by-line to check if they contain the same contents. Used to check the output files for correctness.
bool checkCorrectness(std::string fileAName, std::string fileBName) {
    std::ifstream fileA(fileAName, std::ifstream::in);
    if(!fileA.good()) {
        printf("Error: failed to read from output file %s\n", fileAName.c_str());
        return false;
    }
    std::ifstream fileB(fileBName, std::ifstream::in);
    if(!fileB.good()) {
        printf("Error: failed to read from output file %s\n", fileBName.c_str());
        fileA.close();
        return false;
    }
    std::string lineA;
    std::string lineB;
    while(getline(fileA, lineA)) {
        getline(fileB, lineB);
        if(lineA.compare(lineB) != 0) {
            return false;
        }
    }
    fileA.close();
    fileB.close();
    return true;
}

void dijkstraBenchmark(graph g) {
    Timer t;
    std::vector<float> distance(g.numVertices, INFINITY);
    std::vector<int> predecessor(g.numVertices, -1);
    Dijkstra solver;
    t.reset();
    solver.dijkstra(0, g.vertices, distance, predecessor);
    double elapsed = t.elapsed();
    printf("Dijkstra Runtime: %.4f\n", elapsed);
    saveResults(distance, "out-ref.txt");
}

// Main testing function
int main(int argc, const char **argv) {
    // Parse commandline arguments, reject if no argument given (should give exactly one, which is a file name)
    if(argc != 2) {
        printf("Incorrect number of arguments (%d received, 1 expected)\n", argc - 1);
        return(1);
    }
    // Open file
    std::ifstream file(argv[1], std::ifstream::in);
    if(!file.good()) {
        printf("Input file does not exist\n");
        return(1);
    }
    std::string currentLine;
    graph g;
    getline(file, currentLine);
    g.numVertices = std::stoi(currentLine);
    // Add vertices to graph
    for(int i = 0; i < g.numVertices; i++) {
        std::vector<edge> vec;
        g.vertices.push_back(vec);
    }
    // Iterate through input file and add edges to graph
    while(getline(file, currentLine)) {
        edge e;
        char * str = (char *) currentLine.c_str();
        str = strtok(str, ",");
        int src = std::stoi(str);
        str = strtok(NULL, ",");
        int dest = std::stoi(str);
        str = strtok(NULL, ",");
        float weight = std::stof(str);
        edge newEdge;
        newEdge.dest = dest;
        newEdge.weight = weight;
        g.vertices[src].push_back(newEdge);
    }
    file.close();
    // Testing
    /*
    Timer t;
    t.reset();
    std::vector<float> output;
    // Call algorithm here
    double elapsed = t.elapsed();
    printf("Test Graph Loading: %.4f\n", elapsed);
    saveResults(output, "out.txt");
    bool correct = checkCorrectness("out.txt", "ref.txt"); // Replace ref.txt with name of reference solution
    // Note: Reference solution will probably have to be dijkstra's output
    if(correct) {
        printf("Output correct\n");
    } else {
        printf("Output incorrect\n");
    }*/
    dijkstraBenchmark(g);
    return(0);
}