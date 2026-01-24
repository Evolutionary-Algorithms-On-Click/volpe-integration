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
# {{USER_CODE_INJECTION}}
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