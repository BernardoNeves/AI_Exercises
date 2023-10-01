"""
    @file exercise_DFS-BFS_class-04.py
    @author Bernardo Neves (a23494@alunos.ipca.pt)
    @brief Depth First Search (DFS) and Breadth First Search (BFS) implementation
    @date 2023-09-27
"""

import random, networkx as nx, matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Graph Initialization

    def dfs(self, node, visited):
        if node not in visited:  # If node hasn't been visited
            print(" -> " + node, end="")
            visited.append(node)  # Mark node as visited
            for neighbour in self.graph[node]:
                self.dfs(neighbour, visited)  # Recursevly visit unvisited neighbour

    def bfs(self, initial_node, visited):
        queue = []  # Queue initialization
        visited.append(initial_node)  # Define an initial node, marking it as visited
        queue.append(initial_node)  # Put it on the queue

        while queue:  # As long as the queue is not empty
            node = queue.pop(0)  # Remove the 1st node from the queue
            print(" -> " + node, end="")
            for neighbour in self.graph[node]:  # For each neighbour of node
                if neighbour not in visited:  # If neighbour is not visited
                    visited.append(neighbour)  # Mark neighbour as visited
                    queue.append(neighbour)  # Put neighbour at the end of the queue
                    print(" - *" + neighbour + "* ", end="")

    def get_unvisited_nodes(self, visited):
        return list(self.graph.nodes - visited)  # return unvisited nodes

    def visualize_graph(self):
        G = nx.DiGraph(self.graph)

        pos = nx.spring_layout(G, k=1, seed=2) # Define a layout for the nodes

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=800,
            node_color="skyblue",
            font_size=12,
            font_color="black",
        ) # Draw nodes

        nx.draw_networkx_edges(G, pos, width=1, edge_color="black") # Draw edges
        
        plt.show() # Display the graph

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
    G = Graph()

    for i in range(10):
        G.graph.add_node(f"V{i}") # add i nodes

    nodes = list(G.graph.nodes) + [None]
    for node in G.graph.nodes: # for every node 
        random_node = random.choice(nodes)
        if random_node and random_node != node: # avoid self edges and None
            G.graph.add_edge(node, random_node) # add edge between nodes

    G.run() # run
    G.visualize_graph() # draw graph
