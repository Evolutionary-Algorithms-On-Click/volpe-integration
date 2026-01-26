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
# {{USER_CODE_INJECTION}}
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
                # 1. Select
                offspring = toolbox.select(self.popln, len(self.popln))
                # 2. Clone
                offspring = list(map(toolbox.clone, offspring))
                
                # 3. Mate
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CX_PROB:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # 4. Mutate
                for mutant in offspring:
                    if random.random() < MUT_PROB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                        
                # 5. Evaluate
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    
                # 6. Replace
                self.popln = offspring
                
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
