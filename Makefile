
.phony: all

all:
	g++ -Wall src/bellman.cpp -o bellman
	g++ -Wall src/dijkstra.cpp -o dijkstra
