import multiprocessing
import os
import concurrent.futures
import grpc
import numpy as np
import threading
import sys
import argparse
from typing import Any, override

# Import protos
import volpe_container_pb2 as pb
import common_pb2 as pbc
import volpe_container_pb2_grpc as vp

# --- USER CODE SECTION START ---
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# User and notebook identifiers
USER_ID = "user1"
NOTEBOOK_ID = "tsp_data_001"

# Evolutionary algorithm parameters
POP_SIZE = 10
CX_PROB = 0.8
MUT_PROB = 0.3
N_GENERATIONS = 500

# Problem definition
DIMENSIONS = 1000          # number of cities
LOWER_BOUND = 0          # not used for permutation representation
UPPER_BOUND = 1          # not used for permutation representation

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Load city coordinates
city_df = pd.read_csv("data.csv", header=0)  # assumes columns x and y
city_coords = city_df[["x", "y"]].to_numpy()

def evaluate(individual):
    """Calculate total Euclidean tour length for a permutation."""
    distance = 0.0
    for i in range(len(individual)):
        city_a = city_coords[individual[i]]
        city_b = city_coords[individual[(i + 1) % len(individual)]]
        distance += np.linalg.norm(city_a - city_b)
    return (distance,)

def mate(ind1, ind2):
    """Ordered crossover (OX) for permutations."""
    return tools.cxOrdered(ind1, ind2)

def mutate(individual):
    """Swap mutation for permutations."""
    return tools.mutShuffleIndexes(individual, indpb=0.5 / DIMENSIONS)

def select(population, k):
    """Tournament selection."""
    return tools.selTournament(population, k, tournsize=3)

# Additional helper: evaluate population fitness in one call
def evaluate_population(pop):
    fitnesses = list(map(evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

def create_individual():
    """Create a random permutation representing a tour."""
    return creator.Individual(random.sample(range(DIMENSIONS), DIMENSIONS))

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", select)
# --- USER CODE SECTION END ---


# --- SERIALIZATION HELPERS ---

def individual_to_bytes(ind):
    """Encodes a DEAP individual to bytes (float32 buffer)."""
    # Flattens list/array to float32 bytes.
    # If your individual is complex (e.g. tree), you must adapt this logic.
    try:
        arr = np.array(ind, dtype=np.float32)
        return pbc.Individual(genotype=arr.tobytes(), fitness=ind.fitness.values[0] if ind.fitness.valid else 0.0)
    except Exception as e:
        print(f"Serialization Error: {e}", file=sys.stderr)
        return pbc.Individual(genotype=b'', fitness=0.0)

def bytes_to_individual(pb_ind):
    """Decodes a protobuf Individual back to a DEAP creator.Individual object."""
    # Reads bytes as float32
    arr = np.frombuffer(pb_ind.genotype, dtype=np.float32)
    
    # CONVERSION NOTICE:
    # If your problem uses Integers (like TSP), cast here: arr.astype(int).tolist()
    # If your problem uses Floats, keep as is: arr.tolist()
    # Defaulting to float for generic template, CHANGE IF NEEDED:
    native_data = arr.tolist() 
    
    ind = creator.Individual(native_data)
    ind.fitness.values = (pb_ind.fitness,)
    return ind

def individual_to_string(ind):
    """Encodes a DEAP individual to a string representation for display."""
    return pb.ResultIndividual(representation=str(ind), 
                               fitness=ind.fitness.values[0] if ind.fitness.valid else 0.0)

# --- SERVICE CLASS ---

class VolpeGreeterServicer(vp.VolpeContainerServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popln = []
        self.poplock = threading.Lock()
        
        # Try to initialize if toolbox is ready
        try:
            self._init_population()
        except Exception as e:
            print(f"Warning: Initial population creation failed (check toolbox setup): {e}")

    def _init_population(self):
        self.popln = toolbox.population(n=POP_SIZE)
        fitnesses = list(map(toolbox.evaluate, self.popln))
        for ind, fit in zip(self.popln, fitnesses):
            ind.fitness.values = fit

    @override
    def SayHello(self, request: pb.HelloRequest, context: grpc.ServicerContext):
        return pb.HelloReply(message="Volpe Service Ready")
    
    @override
    def InitFromSeed(self, request: pb.Seed, context: grpc.ServicerContext):
        with self.poplock:
            self._init_population()
            return pb.Reply(success=True)
            
    @override
    def InitFromSeedPopulation(self, request: pbc.Population, context: grpc.ServicerContext):
        with self.poplock:
            incoming_pop = [bytes_to_individual(ind) for ind in request.members]
            if not self.popln:
                 self.popln = incoming_pop
            else:
                self.popln.extend(incoming_pop)
                
            # Maintain size
            if len(self.popln) > POP_SIZE:
                 self.popln = toolbox.select(self.popln, POP_SIZE)
            return pb.Reply(success=True)
            
    @override
    def GetBestPopulation(self, request: pb.PopulationSize, context):
        with self.poplock:
            if not self.popln:
                return pbc.Population(members=[], problemID=NOTEBOOK_ID)
            
            # Use selBest which respects the fitness weights (min or max)
            best_inds = tools.selBest(self.popln, k=min(len(self.popln), request.size))
            pb_members = [individual_to_bytes(ind) for ind in best_inds]
            return pbc.Population(members=pb_members, problemID=NOTEBOOK_ID)
            
    @override
    def GetResults(self, request: pb.PopulationSize, context):
        with self.poplock:
            if not self.popln:
                return pb.ResultPopulation(members=[])
            best_inds = tools.selBest(self.popln, k=min(len(self.popln), request.size))
            res_members = [individual_to_string(ind) for ind in best_inds]
            return pb.ResultPopulation(members=res_members)
            
    @override
    def GetRandom(self, request: pb.PopulationSize, context):
        with self.poplock:
            if not self.popln:
                 return pbc.Population(members=[], problemID=NOTEBOOK_ID)
            k = min(len(self.popln), request.size)
            selected = random.sample(self.popln, k)
            pb_members = [individual_to_bytes(ind) for ind in selected]
            return pbc.Population(members=pb_members, problemID=NOTEBOOK_ID)
            
    @override
    def AdjustPopulationSize(self, request: pb.PopulationSize, context: grpc.ServicerContext):
        with self.poplock:
             target_size = request.size
             current_size = len(self.popln)
             
             if current_size < target_size:
                 needed = target_size - current_size
                 new_inds = toolbox.population(n=needed)
                 fitnesses = map(toolbox.evaluate, new_inds)
                 for ind, fit in zip(new_inds, fitnesses):
                     ind.fitness.values = fit
                 self.popln.extend(new_inds)
             elif current_size > target_size:
                 self.popln = tools.selBest(self.popln, target_size)
             return pb.Reply(success=True)
             
    @override
    def RunForGenerations(self, request: pb.PopulationSize, context):
        generations = request.size if request.size > 0 else 1
        with self.poplock:
            for _ in range(generations):
                # 1. Identify Elite (Best Parent)
                # We keep a reference to the absolute best parent
                elite = tools.selBest(self.popln, 1)[0]
                elite = toolbox.clone(elite) # Clone it so it remains safe

                # 2. Select Parents for Next Gen
                # This uses your Tournament selection (which preserves diversity)
                offspring = toolbox.select(self.popln, len(self.popln))
                offspring = list(map(toolbox.clone, offspring))
                
                # 3. Apply Crossover (Mate)
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CX_PROB:
                        toolbox.mate(child1, child2)
                        if hasattr(child1.fitness, 'values'): del child1.fitness.values
                        if hasattr(child2.fitness, 'values'): del child2.fitness.values
                
                # 4. Apply Mutation
                for mutant in offspring:
                    if random.random() < MUT_PROB:
                        toolbox.mutate(mutant)
                        if hasattr(mutant.fitness, 'values'): del mutant.fitness.values
                        
                # 5. Evaluate Invalid Individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                # Fallback to global evaluate if toolbox fails
                eval_func = getattr(toolbox, 'evaluate', globals().get('evaluate'))
                
                if eval_func:
                    fitnesses = map(eval_func, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                # 6. REPLACEMENT STRATEGY (The Fix)
                # Instead of keeping the 'Best of All', we keep ALL children.
                # This allows 'worse' individuals to survive, maintaining diversity.
                self.popln = offspring
                
                # 7. RE-INJECT ELITE
                # We find the WORST child and replace it with the BEST parent.
                # This ensures we never regress, but we don't crowd the population.
                
                # Find index of worst child
                # Check optimization direction (Min or Max)
                is_min = True
                if hasattr(creator, "FitnessMin") and creator.FitnessMin.weights[0] > 0:
                     is_min = False
                
                if is_min:
                    # For minimization, worst is the one with highest fitness
                    worst_idx = max(range(len(self.popln)), key=lambda i: self.popln[i].fitness.values[0])
                else:
                    # For maximization, worst is the one with lowest fitness
                    worst_idx = min(range(len(self.popln)), key=lambda i: self.popln[i].fitness.values[0])
                
                self.popln[worst_idx] = elite

                # 8. DIVERSITY MAINTENANCE
                # Every 50 generations, replace 10% worst with random new individuals
                if _ % 50 == 0 and _ > 0:
                    num_to_replace = max(1, len(self.popln) // 10)
                    worst_indices = sorted(range(len(self.popln)), 
                                        key=lambda i: self.popln[i].fitness.values[0], 
                                        reverse=True)[:num_to_replace]
                    
                    new_inds = toolbox.population(n=num_to_replace)
                    fitnesses = map(eval_func, new_inds)
                    for ind, fit in zip(new_inds, fitnesses):
                        ind.fitness.values = fit
                    
                    for idx, new_ind in zip(worst_indices, new_inds):
                        self.popln[idx] = new_ind
                                
        return pb.Reply(success=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='8081', help='Port to listen on')
    args, unknown = parser.parse_known_args()

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    vp.add_VolpeContainerServicer_to_server(VolpeGreeterServicer(), server)
    
    print(f"Universal DEAP Service starting on port {args.port}...")
    server.add_insecure_port(f"0.0.0.0:{args.port}")
    
    server.start()
    sys.stdout.flush()
    server.wait_for_termination()
