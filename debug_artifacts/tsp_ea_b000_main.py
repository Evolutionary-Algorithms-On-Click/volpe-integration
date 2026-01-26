# --- IMPORTS START ---
import multiprocessing
import os
import concurrent.futures
import grpc
import numpy as np
import threading
import sys
import random
import dill
import argparse
from typing import Any, override

# Anti-blocking for Matplotlib
try:
    import matplotlib
    matplotlib.use('Agg') # Force non-interactive backend
    import matplotlib.pyplot as plt
    # Mock show() to prevent blocking
    plt.show = lambda *args, **kwargs: None
except ImportError:
    pass

# Import protos
import volpe_container_pb2 as pb
import common_pb2 as pbc
import volpe_container_pb2_grpc as vp
# --- IMPORTS END ---

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
# population, logbook, hall_of_fame = main()  [COMMENTED OUT BY BUILDER]

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
#     plt.show()  [COMMENTED OUT BY BUILDER]

# plot_tour(best_ind, CITIES)  [COMMENTED OUT BY BUILDER]
# --- USER CODE SECTION END ---

# --- STATIC INJECTION HYDRATION START ---
# Try to load pre-calculated static context (CITIES, GRAPH, etc.)
# This ensures all workers have the exact same problem definition
# even if we reset the random seeds later.
try:
    if os.path.exists("static_context.dill"):
        with open("static_context.dill", "rb") as f:
            static_data = dill.load(f)
            # Inject into global scope
            # We explicitly overwrite any variables the user code might have set
            # to ensure consistency with the build-time snapshot.
            globals().update(static_data)
        print(f"Successfully injected {len(static_data)} static variables from build context.")
except Exception as e:
    print(f"WARNING: Failed to load static_context.dill: {e}")
# --- STATIC INJECTION HYDRATION END ---

# --- SEED RESET START ---
# Now that static data (Problem) is loaded, we MUST reset the seeds 
# so that the solver (Population) evolves differently on each worker.
# This overrides any specific seed set in the user code section above.
random.seed(None)
np.random.seed(None)
# --- SEED RESET END ---

# --- ADAPTERS START ---
# These adapter functions bridge the gap between user-defined functions (which might use DEAP objects)
# and the Volpe Service (which expects (numpy_array, fitness_float) tuples).

def adapter_gen_ind():
    """Generates an individual using user's gen_ind or DEAP toolbox, returns (np.array, float)."""
    # 1. Try 'adapter_gen_ind' if user defined it (explicit adapter)
    if 'adapter_gen_ind' in globals() and globals()['adapter_gen_ind'] != adapter_gen_ind: 
         # Prevent recursion if user defines same name, but usually they won't.
         # This block is for if the user code ITSELF defined an adapter.
         return globals()['adapter_gen_ind']()

    # 2. Try DEAP toolbox (common case)
    if 'toolbox' in globals() and hasattr(globals()['toolbox'], 'individual'):
        ind = globals()['toolbox'].individual()
        # Evaluate if not evaluated
        if not ind.fitness.valid:
            fit = globals()['toolbox'].evaluate(ind)
            ind.fitness.values = fit
        
        # Convert to numpy
        # If it's a list (permutation), standard conversion
        return (np.array(ind, dtype=np.float32), float(ind.fitness.values[0]))

    # 3. Try generic 'gen_ind' from user (e.g. simple opfunu code)
    if 'gen_ind' in globals():
        res = globals()['gen_ind']()
        # Ensure it returns tuple (ind, fit)
        if isinstance(res, tuple) and len(res) == 2:
            return res
        # If it returns just ind, we might need to evaluate?
        # Assuming user follows protocol
        return res
    
    raise NotImplementedError("No suitable individual generator found (checked toolbox.individual and gen_ind)")

def adapter_crossover(tup1, tup2):
    # 1. Try DEAP toolbox
    if 'toolbox' in globals() and hasattr(globals()['toolbox'], 'mate') and 'creator' in globals():
        creator = globals()['creator']
        toolbox = globals()['toolbox']
        
        # Reconstruct DEAP individuals
        # We assume 'Individual' is the name, or we grab the first class in creator??
        # Usually creator.Individual is standard.
        if hasattr(creator, 'Individual'):
            ind1 = creator.Individual(tup1[0].astype(int).tolist()) # int for permutation? or float?
            ind2 = creator.Individual(tup2[0].astype(int).tolist())
            
            # For float problems (not TSP), casting to int is bad.
            # We need to detect if integer or float.
            # Heuristic: Check dtype of input tuple
            if np.issubdtype(tup1[0].dtype, np.floating) and not np.all(np.mod(tup1[0], 1) == 0):
                 ind1 = creator.Individual(tup1[0].tolist())
                 ind2 = creator.Individual(tup2[0].tolist())
            
            ind1.fitness.values = (tup1[1],)
            ind2.fitness.values = (tup2[1],)
            
            child1, child2 = toolbox.mate(ind1, ind2)
            
            # Re-eval
            if hasattr(toolbox, 'evaluate'):
                # Invalidate
                del child1.fitness.values
                del child2.fitness.values
                
                f1 = toolbox.evaluate(child1)
                f2 = toolbox.evaluate(child2)
                
                return (np.array(child1, dtype=np.float32), float(f1[0])), (np.array(child2, dtype=np.float32), float(f2[0]))
    
    # 2. Try generic 'crossover' function
    if 'crossover' in globals():
        return globals()['crossover'](tup1, tup2)

    return tup1, tup2 # No op

def adapter_mutate(tup):
    # 1. Try DEAP toolbox
    if 'toolbox' in globals() and hasattr(globals()['toolbox'], 'mutate') and 'creator' in globals():
        creator = globals()['creator']
        toolbox = globals()['toolbox']
        
        if hasattr(creator, 'Individual'):
             # Detect type again
             if np.issubdtype(tup[0].dtype, np.floating) and not np.all(np.mod(tup[0], 1) == 0):
                 ind = creator.Individual(tup[0].tolist())
             else:
                 ind = creator.Individual(tup[0].astype(int).tolist())
             
             ind.fitness.values = (tup[1],)
             
             mutated, = toolbox.mutate(ind)
             
             # Re-eval
             if hasattr(toolbox, 'evaluate'):
                 f = toolbox.evaluate(mutated)
                 return (np.array(mutated, dtype=np.float32), float(f[0]))

    # 2. Try generic 'mutate'
    if 'mutate' in globals():
        return globals()['mutate'](tup)
        
    return tup

def adapter_select(popln_tuples, k):
    # 1. Try DEAP toolbox
    # DEAP select usually works on objects with .fitness. 
    # Converting entire population to DEAP objects just to select might be slow.
    # But if the user defined a specific selection operator (like Tournament), we should use it?
    # Actually, simpler to just use generic tournament on the tuples if user didn't define 'select'.
    
    if 'select' in globals():
        # If user defined 'select', it might expect DEAP objects or Tuples?
        # If it's DEAP tools.selTournament, it expects Objects.
        # We can't easily use DEAP 'select' on tuples.
        pass 

    # Default: Robust Tournament Selection on Tuples
    # This works for any representation (tuple based)
    TOURNAMENT_SIZE = globals().get('TOURNAMENT_SIZE', 3)
    chosen = []
    for _ in range(k):
        candidates = [random.choice(popln_tuples) for _ in range(TOURNAMENT_SIZE)]
        # Min minimization assumed? Or check user weights?
        # DEAP: creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # If weights are negative -> Min. 
        # We'll assume Minimization (fitness[0]) or user can sort.
        best = min(candidates, key=lambda x: x[1])
        chosen.append(best)
    return chosen

# --- ADAPTERS END ---

# --- SERVICE HELPER FUNCTIONS ---

def popListTostring(popln: list[tuple[np.ndarray, float]]):
    indList : list[pb.ResultIndividual] = []
    for mem in popln:
        # Convert floats back to ints for display if it looks like int
        if np.all(np.mod(mem[0], 1) == 0):
             clean_repr = str(mem[0].astype(int).tolist())
        else:
             clean_repr = str(mem[0].tolist())
             
        indList.append(
                pb.ResultIndividual(representation=clean_repr, 
                                    fitness=mem[1])
                )
    return pb.ResultPopulation(members=indList)

def bstringToPopln(popln: pbc.Population):
    popList = []
    for memb in popln.members:
        popList.append((np.frombuffer(memb.genotype, dtype=np.float32), memb.fitness))
    return popList

def popListToBytes(popln: list[tuple[np.ndarray, float]]):
    indList : list[pbc.Individual] = []
    for mem in popln:
        indList.append(pbc.Individual(genotype=mem[0].astype(np.float32).tobytes(), fitness=mem[1]))
    return pbc.Population(members=indList, problemID="tsp_p1")

def expand(popln, newPop):
    if len(popln) == 0:
        return [adapter_gen_ind() for _ in range(newPop)]
    needed = newPop - len(popln)
    if needed <= 0: return popln
    
    extras = []
    while len(extras) < needed:
        p1 = random.choice(popln)
        p2 = random.choice(popln)
        c1, c2 = adapter_crossover(p1, p2)
        extras.append(c1)
        if len(extras) < needed:
            extras.append(c2)
    return popln + extras

# --- SERVICE CLASS ---

class VolpeGreeterServicer(vp.VolpeContainerServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poplock = threading.Lock()
        
        # Initialize population using Adapter
        # Determine size
        pop_size = globals().get('POP_SIZE', globals().get('BASE_POPULATION_SIZE', 50))
        
        print(f"Initializing population of size {pop_size}...")
        try:
            self.popln = [ adapter_gen_ind() for _ in range(pop_size) ]
        except Exception as e:
            print(f"Error initializing population: {e}")
            self.popln = []

    @override
    def SayHello(self, request: pb.HelloRequest, context: grpc.ServicerContext):
        return pb.HelloReply(message="hello " + request.name)
    
    @override
    def InitFromSeed(self, request: pb.Seed, context: grpc.ServicerContext):
        with self.poplock:
            pop_size = globals().get('POP_SIZE', globals().get('BASE_POPULATION_SIZE', 50))
            self.popln = [ adapter_gen_ind() for _ in range(pop_size) ]
            return pb.Reply(success=True)
            
    @override
    def InitFromSeedPopulation(self, request: pbc.Population, context: grpc.ServicerContext):
        with self.poplock:
            ogLen = len(self.popln)
            seedPop = bstringToPopln(request)
            if self.popln is None:
                self.popln = []
            self.popln.extend(seedPop)
            self.popln = adapter_select(self.popln, ogLen)
            return pb.Reply(success=True)
            
    @override
    def GetBestPopulation(self, request: pb.PopulationSize, context):
        with self.poplock:
            if not self.popln:
                return pbc.Population(members=[], problemID="tsp_p1")
            popSorted = sorted(self.popln, key=lambda x: x[1])
            return popListToBytes(popSorted[:request.size])
            
    @override
    def GetResults(self, request: pb.PopulationSize, context):
        with self.poplock:
            if not self.popln:
                return pb.ResultPopulation(members=[])
            popSorted = sorted(self.popln, key=lambda x: x[1])
            return popListTostring(popSorted[:request.size])
            
    @override
    def GetRandom(self, request: pb.PopulationSize, context):
        with self.poplock:
            if not self.popln:
                return pbc.Population(members=[], problemID="tsp_p1")
            indices = np.random.randint(0, len(self.popln), request.size)
            popList = [self.popln[i] for i in indices]
            return popListToBytes(popList)
            
    @override
    def AdjustPopulationSize(self, request: pb.PopulationSize, context: grpc.ServicerContext):
        with self.poplock:
             if len(self.popln) < request.size:
                 self.popln = expand(self.popln, request.size)
             elif len(self.popln) > request.size:
                 self.popln = adapter_select(self.popln, request.size)
             return pb.Reply(success=True)
             
    @override
    def RunForGenerations(self, request: pb.PopulationSize, context):
        with self.poplock:
            ogLen = len(self.popln)
            # 1. Select
            parents = adapter_select(self.popln, ogLen)
            
            newpopln = []
            # 2. Crossover
            CX_PROB = globals().get('CX_PROB', 0.8)
            for i in range(0, ogLen, 2):
                if i+1 < len(parents) and np.random.random() < CX_PROB:
                    child1, child2 = adapter_crossover(parents[i], parents[i+1])
                    newpopln.append(child1)
                    newpopln.append(child2)
                else:
                    newpopln.append(parents[i])
                    if i+1 < len(parents):
                        newpopln.append(parents[i+1])
            
            # 3. Mutation
            MUT_PROB = globals().get('MUT_PROB', 0.2)
            for i in range(len(newpopln)):
                if np.random.random() < MUT_PROB:
                    newpopln[i] = adapter_mutate(newpopln[i])
            
            self.popln = newpopln
            
            # Ensure size consistency
            if len(self.popln) != ogLen:
                self.popln = adapter_select(self.popln, ogLen)
                
        return pb.Reply(success=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='8081', help='Port to listen on')
    args, unknown = parser.parse_known_args()

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    vp.add_VolpeContainerServicer_to_server(VolpeGreeterServicer(), server)
    
    print(f"Server starting on port {args.port}...")
    server.add_insecure_port(f"0.0.0.0:{args.port}")
    
    server.start()
    sys.stdout.flush()
    server.wait_for_termination()
