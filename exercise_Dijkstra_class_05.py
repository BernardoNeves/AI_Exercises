"""
    @file exercise_Dijkstra_class_05.py
    @author Bernardo Neves (a23494@alunos.ipca.pt)
    @brief Dijkstra implementation
    @date 2023-10-01
"""

import sys

import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Graph Initialization

    def dijkstra(self, initial_node):
        opened = [node for node in self.graph.nodes]

        while initial_node in opened: # while U node is open
            print(f"U: {initial_node},\t\t Parent: {self.graph.nodes[initial_node]['parent']},\t Weight: {self.graph.nodes[initial_node]['weight']}")
            for neighbour in self.graph.neighbors(initial_node): # For each V neighbour of node
                if neighbour in opened: # If V neighbour is open
                    edge_weight = self.graph.get_edge_data(initial_node, neighbour)["weight"]
                    parent_weight = self.graph.nodes[initial_node]["weight"]
                    new_weight = edge_weight + parent_weight
                    neighbour_weight = self.graph.nodes[neighbour]["weight"]
                    if new_weight < neighbour_weight: # if the weight of the new path is less heavy than the V neighbour's weight (previous path)
                        self.graph.nodes[neighbour].update({"parent": initial_node, "weight": new_weight}) # update the V neighbour's parent and weight based on the new shortest path
                    print(f" - V: {neighbour},\t Parent: {self.graph.nodes[neighbour]['parent']},\t Weight: {self.graph.nodes[neighbour]['weight']}")
            print(f"U: {initial_node} CLOSED,\t Parent: {self.graph.nodes[initial_node]['parent']},\t Weight: {self.graph.nodes[initial_node]['weight']}\n")
            opened.remove(initial_node) # close node since it doesn't have any neighbours remaining
            
            for node in opened: # for every open node 
                node_weight = self.graph.nodes[node]["weight"]
                initial_node_weight = self.graph.nodes[initial_node]["weight"]
                if node_weight < initial_node_weight or initial_node not in opened: # find lowest weight node 
                    initial_node = node # set it as U node

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
        print("\n> Dijkstra:")
        self.dijkstra(initial_node) # run Dijkstra

        parent = goal_node
        path = []

        while parent != -1: # while node has a parent
            node_info = f"{parent} W:{self.graph.nodes[parent]['weight']}"  # get parent info
            path.append(node_info) # append it to the path
            parent = self.graph.nodes[parent]["parent"] # get it's parent

        path_str = " -> ".join(reversed(path)) # get path string
        print(f"\nDijkstra Path: {path_str}")
        

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
    goal_node = "V5" # set our goal node

    
    G.run(initial_node, goal_node) # runtime
    G.visualize_graph() # draw graph
