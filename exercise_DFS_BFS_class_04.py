"""
    @file exercise_DFS-BFS_class-04.py
    @author Bernardo Neves (a23494@alunos.ipca.pt)
    @brief Depth First Search (DFS) and Breadth First Search (BFS) implementation
    @date 2023-09-27
"""

import random

class Graph:
    def __init__(self):
        self.graph = {
            "V0": ["V1", "V7"],
            "V1": ["V4", "V6"],
            "V2": ["V0"],
            "V3": ["V4", "V5"],
            "V4": [],
            "V5": [],
            "V6": [],
            "V7": ["V2"],
            "V8": ["V1"],
            "V9": ["V3", "V5", "V8"],
        }  # Graph Initialization

    def dfs(self, node, visited):
        if node not in visited:  # If node hasn't been visited
            print(" -> " + node, end="")
            visited.append(node)  # Mark node as visited
            for neighbor in self.graph[node]:  # For each neighbour of node
                self.dfs(neighbor, visited)  # Recursevly visit unvisited neighbour

    def bfs(self, initial_node, visited):
        queue = []  # Queue initialization
        visited.append(initial_node)  # Define an initial node, marking it as visited
        queue.append(initial_node)  # Put it on the queue

        while queue:  # As long as the queue is not empty
            node = queue.pop(0)  # Remove the 1st node from the queue
            print(" -> " + node, end="")
            for neighbor in self.graph[node]:  # For each neighbour of node
                if neighbor not in visited:  # If neighbour is not visited
                    visited.append(neighbor)  # Mark neighbour as visited
                    queue.append(neighbor)  # Put neighbour at the end of the queue
                    print(" - *" + neighbor + "* ", end="")

    def get_unvisited_nodes(self, visited):
        return list(set(self.graph.keys()) - set(visited))  # return unvisited nodes

    def run(self):
        visited = []  # Visited list initialization
        unvisited = self.get_unvisited_nodes(visited)  # get unvisited nodes
        print("\n> DFS:")

        while unvisited:  # While there are unvisited nodes
            print("\n\t> Start", end="")
            node = random.choice(unvisited)  # Randomly pick an unvisited neighbour
            self.dfs(node, visited)  # Start a Depth First Search from this node
            unvisited = self.get_unvisited_nodes(visited)  # get unvisited nodes
        print("\n\nDFS Visited: " + str(visited) + "\n")

        visited.clear()  # Empty visited list
        unvisited = self.get_unvisited_nodes(visited)  # get unvisited nodes
        print("\n> BFS:")

        while unvisited:  # While there are unvisited nodes
            print("\n\t> Start", end="")
            node = random.choice(unvisited)  # Randomly pick an unvisited neighbour
            self.bfs(node, visited)  # Start a Breadth First Search from this node
            unvisited = self.get_unvisited_nodes(visited)  # get unvisited nodes
        print("\n\nBFS Visited: " + str(visited) + "\n")


if __name__ == "__main__":
    graph = Graph()
    graph.run()
