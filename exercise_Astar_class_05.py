"""
    @file exercise_DFS-BFS_class-04.py
    @author Bernardo Neves (a23494@alunos.ipca.pt)
    @brief Depth First Search (DFS) and Breadth First Search (BFS) implementation
    @date 2023-09-27
"""

import random, sys, networkx as nx, matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Graph Initialization

    def astar(self, initial_node, goal_node):
        opened = [node for node in self.graph.nodes]
        estimated = {node: self.heuristic(node, goal_node) for node in self.graph.nodes} # create dict containing nodes and their estimated weight to reach the goal node

        while initial_node in opened: # while U node is open
            print(f"U: {initial_node},\t\t Parent: {self.graph.nodes[initial_node]['parent']},\t Weight: {self.graph.nodes[initial_node]['weight']} + {estimated[initial_node]}")
            for neighbour in self.graph.neighbors(initial_node): # For each V neighbour of node
                if neighbour in opened: # If V neighbour is open
                    edge_weight = self.graph.get_edge_data(initial_node, neighbour)["weight"]
                    parent_weight = self.graph.nodes[initial_node]["weight"]
                    new_weight = edge_weight + parent_weight
                    neighbour_weight = self.graph.nodes[neighbour]["weight"]
                    if new_weight < neighbour_weight: # if the weight of the new path is less heavy than the V neighbour's weight (previous path)
                        self.graph.nodes[neighbour].update({"parent": initial_node, "weight": new_weight}) # update the V neighbour's parent and weight based on the new shortest path
                    print(f" - V: {neighbour},\t Parent: {self.graph.nodes[neighbour]['parent']},\t Weight: {self.graph.nodes[neighbour]['weight']} + {estimated[neighbour]}")
            print(f"U: {initial_node} CLOSED,\t Parent: {self.graph.nodes[initial_node]['parent']},\t Weight: {self.graph.nodes[initial_node]['weight']} + {estimated[initial_node]}\n")
            opened.remove(initial_node) # close node since it doesn't have any neighbours remaining
            if initial_node == goal_node:
                return

            for node in opened: # for every open node 
                estimated_node_weight = self.graph.nodes[node]["weight"] + estimated[node] # use estimation value
                estimated_initial_node_weight = self.graph.nodes[initial_node]["weight"] + estimated[initial_node] # use estimation value
                if estimated_node_weight < estimated_initial_node_weight or initial_node not in opened: # find lowest weight node according to the estimation
                    initial_node = node # set it as U node
    
    def heuristic(self, node, goal_node):
        return (int(goal_node[1]) - int(node[1])) * 10 # returns difference from id's * 10, (('V'5 - 'V'1)* 10 = 40)        

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

    def run(self, initial_node, goal_node):
        print("\n> A*:")
        self.astar(initial_node, goal_node) # run A*

        parent = "V5"
        path = []

        while parent != -1: # while node has a parent
            node_info = f"{parent} W:{self.graph.nodes[parent]['weight']}"  # get parent info
            path.append(node_info) # append it to the path
            parent = self.graph.nodes[parent]["parent"] # get it's parent

        path_str = " -> ".join(reversed(path)) # get path string
        print(f"\nA* Path: {path_str}")
        

if __name__ == "__main__":
    G = Graph()

    nodes_with_attributes = {f"V{i}": {'weight': sys.maxsize, 'parent': -1} for i in range(6)} # create dict containing nodes with default attributes
    G.graph.add_nodes_from(nodes_with_attributes.items()) # add nodes from dict
    # add edges
    G.graph.add_edge("V0", "V1", weight=10)
    G.graph.add_edge("V0", "V2", weight=5)
    G.graph.add_edge("V1", "V3", weight=1)
    G.graph.add_edge("V2", "V1", weight=3)
    G.graph.add_edge("V2", "V3", weight=8)
    G.graph.add_edge("V2", "V4", weight=2)
    G.graph.add_edge("V3", "V4", weight=4)
    G.graph.add_edge("V3", "V5", weight=4)
    G.graph.add_edge("V4", "V5", weight=6)

    initial_node = "V0" # set our initial node
    G.graph.nodes["V0"]["weight"] = 0 # reset initial node weight to 0 since it's the start
    goal_node = "V5"

    G.run(initial_node, goal_node)
    G.visualize_graph() # draw graph
