import multiprocessing
import os
import concurrent.futures
import grpc
import numpy as np
import threading
import sys
from typing import Any

# Import protos
import volpe_container_pb2 as pb
import common_pb2 as pbc
import volpe_container_pb2_grpc as vp

# --- USER CODE SECTION START ---
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Configuration constants
USER_ID = "sudharsan"
NOTEBOOK_ID = "tsp_ea_001"

NUM_CITIES = 10
POP_SIZE = 100
NGEN = 200
CX_PROB = 0.8
MUT_PROB = 0.2
TOURNAMENT_SIZE = 3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Random coordinates for cities in unit square
CITIES = np.random.rand(NUM_CITIES, 2)

# Pre-compute distance matrix
DIST_MATRIX = np.sqrt(((CITIES[:, np.newaxis, :] - CITIES[np.newaxis, :, :]) ** 2).sum(axis=2))

# DEAP creator: fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate(individual):
    """Calculate total tour length for a permutation of city indices."""
    distance = 0.0
    for i in range(len(individual)):
        from_city = individual[i]
        to_city = individual[(i + 1) % len(individual)]
        distance += DIST_MATRIX[from_city, to_city]
    return (distance,)

def mate(ind1, ind2):
    """Ordered crossover (OX) between two individuals."""
    return tools.cxOrdered(ind1, ind2)

def mutate(individual):
    """Swap mutation: exchange two positions."""
    return tools.mutSwap(individual, indpb=1.0/NUM_CITIES)

def select(population, k):
    """Tournament selection."""
    return tools.selTournament(population, k, tournsize=TOURNAMENT_SIZE)

# Additional helper: statistics collector
def get_stats():
    """Create a DEAP Statistics object for fitness."""
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    return stats

def create_individual():
    """Generate a random permutation individual."""
    return creator.Individual(random.sample(range(NUM_CITIES), NUM_CITIES))

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", select)
toolbox.register("map", map)  # use built‑in map for parallelism if needed

def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = get_stats()

    # Evolutionary algorithm (simple EA)
    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                        cxpb=CX_PROB,
                                        mutpb=MUT_PROB,
                                        ngen=NGEN,
                                        stats=stats,
                                        halloffame=hof,
                                        verbose=True)
    return pop, logbook, hof

# Run the algorithm
population, logbook, hall_of_fame = main()

best_ind = hall_of_fame[0]
best_distance = evaluate(best_ind)[0]

print(f"Best tour distance: {best_distance:.4f}")
print(f"Best tour: {best_ind}")

# Plot the best tour
def plot_tour(tour, cities):
    plt.figure(figsize=(8,6))
    # Plot cities
    plt.scatter(cities[:,0], cities[:,1], c='red')
    for i, (x, y) in enumerate(cities):
        plt.text(x, y+0.02, str(i), fontsize=9, ha='center')
    # Create ordered coordinates (return to start)
    ordered = np.append(tour, tour[0])
    plt.plot(cities[ordered,0], cities[ordered,1], 'b-', lw=2)
    plt.title('Best TSP Tour')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

plot_tour(best_ind, CITIES)
# --- USER CODE SECTION END ---

# Defaults if not provided by user
if 'BASE_POPULATION_SIZE' not in globals():
    BASE_POPULATION_SIZE = 100
else:
    BASE_POPULATION_SIZE = globals()['BASE_POPULATION_SIZE']

class VolpeService(vp.VolpeContainerServicer):
    def __init__(self):
        self.poplock = threading.Lock()
        self.ensure_user_functions()
        
        # Initialize population
        # We expect gen_ind to return (individual_representation, fitness) or just representation?
        # The user code in volpe-py returned (ind, fitness). We will assume that protocol.
        self.popln = [self.gen_ind() for _ in range(BASE_POPULATION_SIZE)]

    def ensure_user_functions(self):
        # Map globals to instance methods
        required = ['fitness', 'gen_ind', 'mutate', 'crossover', 'select']
        for name in required:
            if name in globals():
                setattr(self, name, globals()[name])
            else:
                print(f"CRITICAL WARNING: User function '{name}' not found. Service may fail.")
                # We could define dummy fallbacks here if needed
                
    def SayHello(self, request, context):
        return pb.HelloReply(message="Hello " + request.name)

    def InitFromSeed(self, request, context):
        with self.poplock:
            # Re-initialize population
            self.popln = [self.gen_ind() for _ in range(BASE_POPULATION_SIZE)]
            return pb.Reply(success=True)

    def InitFromSeedPopulation(self, request, context):
        with self.poplock:
            ogLen = len(self.popln)
            # Decode incoming population
            seedPop = []
            for memb in request.members:
                # Assuming genotype is bytes, we might need a user-defined decode function
                # or assume standard numpy float32 bytes if not specified.
                # For this template, we'll try to use a 'decode' function if it exists, else generic numpy
                if 'decode' in globals():
                    ind_val = globals()['decode'](memb.genotype)
                else:
                    ind_val = np.frombuffer(memb.genotype, dtype=np.float32)
                
                # Recalculate fitness? Or trust the one sent?
                # Usually we trust the one sent or re-evaluate. 
                # Let's trust it for speed, or re-eval if needed.
                # For consistency with local format (ind, fitness), we store that tuple.
                seedPop.append((ind_val, memb.fitness))

            if self.popln is None:
                self.popln = []
            self.popln.extend(seedPop)
            
            # Select back to size
            self.popln = self.select(self.popln, ogLen)
            return pb.Reply(success=True)

    def GetBestPopulation(self, request, context):
        with self.poplock:
            if not self.popln:
                return pbc.Population(members=[], problemID="p1")
            
            # Sort by fitness (assuming lower is better? Or depends on user problem. 
            # We usually assume minimization or the user sorts in 'select'.)
            # But here we need to return the "Best".
            # We'll assume the user's fitness function returns a float where smaller is better 
            # OR we rely on standard sorting.
            # A safer bet: sort by key=lambda x: x[1] (fitness)
            popSorted = sorted(self.popln, key=lambda x: x[1])
            best_k = popSorted[:request.size]
            
            # Encode back to bytes
            indList = []
            for mem in best_k:
                if 'encode' in globals():
                    gen_bytes = globals()['encode'](mem[0])
                else:
                    gen_bytes = mem[0].astype(np.float32).tobytes()
                
                indList.append(pbc.Individual(genotype=gen_bytes, fitness=mem[1]))
                
            return pbc.Population(members=indList, problemID="p1")

    def GetResults(self, request, context):
        # Similar to GetBestPopulation but returns ResultIndividual (string representation)
        with self.poplock:
            if not self.popln:
                return pb.ResultPopulation(members=[])
            
            popSorted = sorted(self.popln, key=lambda x: x[1])
            best_k = popSorted[:request.size]
            
            indList = []
            for mem in best_k:
                # Use encode_str if available
                if 'encode_str' in globals():
                    rep_str = globals()['encode_str'](mem[0])
                else:
                    rep_str = str(mem[0])
                    
                indList.append(pb.ResultIndividual(representation=rep_str, fitness=mem[1]))
                
            return pb.ResultPopulation(members=indList)

    def GetRandom(self, request, context):
        with self.poplock:
            if not self.popln:
                return pbc.Population(members=[], problemID="p1")
            
            indices = np.random.randint(0, len(self.popln), request.size)
            selected = [self.popln[i] for i in indices]
            
            indList = []
            for mem in selected:
                if 'encode' in globals():
                    gen_bytes = globals()['encode'](mem[0])
                else:
                    gen_bytes = mem[0].astype(np.float32).tobytes()
                indList.append(pbc.Individual(genotype=gen_bytes, fitness=mem[1]))
                
            return pbc.Population(members=indList, problemID="p1")

    def RunForGenerations(self, request, context):
        # request.size is treated as generations or ignore? 
        # The proto name is PopulationSize but context usually implies "steps" or just "run one step".
        # We'll assume one generation step or standard evolution cycle.
        with self.poplock:
            # Standard EA Loop: Select -> Crossover -> Mutate -> Replace
            ogLen = len(self.popln)
            
            # Select parents
            parents = self.select(self.popln, ogLen)
            
            offspring = []
            # Generate offspring (pairwise crossover)
            for i in range(0, ogLen, 2):
                if i+1 < len(parents):
                    if np.random.random() < 0.9: # Crossover prob default
                         child1, child2 = self.crossover(parents[i], parents[i+1])
                         offspring.append(child1)
                         offspring.append(child2)
                    else:
                        offspring.append(parents[i])
                        offspring.append(parents[i+1])
                else:
                    offspring.append(parents[i])
            
            # Mutate
            for i in range(len(offspring)):
                # We assume mutate returns (ind, fitness)
                # Or just modifies ind? User definition varies. 
                # wrapper_main in volpe-py implied: newinds[i] = self.mutate(newinds[i])
                offspring[i] = self.mutate(offspring[i])
                
            # Replace/Merge
            # Simple strategy: Comma or Plus strategy. 
            # We'll assume (mu + lambda) i.e. combine and select best
            combined = self.popln + offspring
            self.popln = self.select(combined, ogLen)
            
            return pb.Reply(success=True)

    def AdjustPopulationSize(self, request, context):
        # Not implemented for now
        return pb.Reply(success=True)

if __name__ == '__main__':
    print("Starting Volpe Container Service...")
    sys.stdout.flush()

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    vp.add_VolpeContainerServicer_to_server(VolpeService(), server)
    server.add_insecure_port("0.0.0.0:8081")
    server.start()
    print("Server listening on 0.0.0.0:8081")
    sys.stdout.flush()
    server.wait_for_termination()