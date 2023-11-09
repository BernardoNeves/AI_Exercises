"""
    @file exercise_Genetic_class_09.py
    @author Bernardo Neves (a23494@alunos.ipca.pt)
    @brief Genetic Algorithm implementation
    @date 2023-11-10
"""

import random as rd
from itertools import product

class ItemType:
    def __init__(self, id, value, weight, max_quantity) -> None:
        self.id = id
        self.value = value
        self.weight = weight
        self.max_quantity = max_quantity
        
def function_objective(individual: list) -> int:
    return sum(item.value * quantity for item, quantity in individual)
    
def get_weight(individual: list) -> int:
    return sum(item.weight * quantity for item, quantity in individual)

def get_universe(items: list, max_weight: int) -> list:
    universe = product(*[range(item.max_quantity + 1) for item in items])
    return [list(zip(items, case)) for case in universe if sum(item.weight * quantity for item, quantity in zip(items, case)) <= max_weight]

def get_population(universe: list, population_size: int, max_weight: int) -> list:
    population = set()
    for _ in range(population_size):
        population.add((chr(65 + len(population)), tuple(rd.choice(universe))))
    return list(population)

def get_mutations(parents: list, mutation_rate: float = 0.1) -> list:
    mutations = []
    for parent in parents:
        if(rd.random() <= mutation_rate):
            mutations.append((parent, rd.choice(parent[1])[0]))
    return mutations
            
def get_offsprings(parents: list, crossover_genes: set, mutations: list, last_letter: str) -> list:
    quantities = [[quantity for _, quantity in parent[1]] for parent in parents]
    offsprings = []
    for i, parent in enumerate(parents):
        quantity = quantities[ i - (len(parents) // 2)] 
        offspring = []
        for item, original_quantity in parent[1]:
            new_quantity = quantity[item.id - 1] if item.id in crossover_genes else original_quantity
            if (parent, item) in mutations:
                new_quantity = rd.randint(0, item.max_quantity)
                print(f"Mutation: Item {item.id}, of {chr(ord(last_letter) + i + 1)}, with quantity {new_quantity}")
            offspring.append((item, new_quantity))
        offsprings.append((chr(ord(last_letter) + i + 1), offspring))
    return offsprings


def print_population(population: list) -> None:
    header = " | ".join(f"Item {i+1}" for i in range(len(population[0][1])))
    print("-" * 50, f"\n   | {header} | Weight | Value")
    for name, individual in population:
        items = " | ".join(f"{quantity:6}" for _, quantity in individual)
        weight = get_weight(individual)
        value = function_objective(individual)
        print(f" {name} | {items} | {weight:6d} | {value:4d}")
    print("-" * 50)


def main() -> None:
    items = [ItemType(id=1, value=40, weight=3, max_quantity=3), ItemType(id=2, value=100, weight=5, max_quantity=2), ItemType(id=3, value=50, weight=2, max_quantity=5)]
    possibilities = get_universe(items, max_weight=20)
    initial_population = get_population(possibilities, population_size=4, max_weight=20)
    print_population(initial_population)
    result = genetic(initial_population, max_weight=20, mutation_rate=0.1, parent_count=2)

    print("\n\t\t Resulting Best:")
    print_population([result])


def genetic(population: list, max_weight: float, mutation_rate: float, parent_count: int = 2) -> list:
    best = None
    last_letter = max([individual[0] for individual in population])
    
    population_count = 0
    while True:
        population_count += 1
        print(f"\n\t\tPopulation {population_count}: ") 
        print_population(sorted(population, key=lambda individual: individual[0]))
        
        population.sort(key=lambda individual: sum([item.value * quantity for item, quantity in individual[1]]), reverse=True)
        if best is None or function_objective(best[1]) < function_objective(population[0][1]):
            best = population[0]
            stale = 0
            print("\n\t\tNew Best:")
            print_population([best])
        else:
            stale += 1
            print(f"\n\tStale: {stale}")
            if stale > 1:
                break
            
        population_length = len(population)
        better_half = population[:population_length // 2]
        worse_half = population[population_length // 2:]
        
        print("\n\t\tBest Individuals:")
        print_population(better_half)
        print("\n\t\tWorst Individuals:")
        print_population(worse_half)
        
        parents = []
        for _ in range(parent_count // 2):
            better_parent = rd.choice(better_half)
            worse_parent = rd.choice(worse_half)
            parents.extend([better_parent, worse_parent])
            better_half.remove(better_parent)
            worse_half.remove(worse_parent)
            
        for parent in parents:
            population.remove(parent)
            
        print("\n\t\tNew PARENTS:")
        print_population(parents)
        
        while True: 
            crossover_genes = set()
            for _ in range(parents.__len__()):
                crossover_genes.add(rd.randint(1, 3))
            print(f"Crossover: {crossover_genes}")               
            
            mutations = get_mutations(parents, mutation_rate)
            offsprings = get_offsprings(parents, crossover_genes, mutations, last_letter)
            
            if all(get_weight(offspring[1]) <= max_weight for offspring in offsprings):
                break
        population.extend(offsprings)
        
        print("\n\t\t NEW CHILDS:")
        print_population(offsprings)   
    return best
    
if __name__ == "__main__":
    main() 