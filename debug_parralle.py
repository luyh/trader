import multiprocessing as mp
import numpy as np
import time,os

NUM_CORES = 2
config = 2

print('Start, Num_Cores:',NUM_CORES)

class Exchange():
    def __init__(self,id,balance):
        self.id = id
        self.balance = balance

class Env():
    def __init__(self,id,exchange):
        self.id = id
        self.exchange = exchange

    def step(self):
        self.exchange.balance *= 2
        time.sleep(1)

def eval_genome(env, genome, config):
    fitness = genome.id
    env.step()
    genome.fitness = fitness + config
    print('run current sub_process %s ,env id:%d,balance:%d,geome id:%d,fitness %d'
          % (os.getpid(),env.id,env.exchange.balance,genome.id,genome.fitness))

    time.sleep(2)
    return fitness

class PooledEvaluate():
    def __init__(self,NUM_CORES,env):
        self.pool = None if NUM_CORES < 2 else mp.Pool(NUM_CORES)
        self.env_array = mp.Array( 'b', np.zeros( NUM_CORES, dtype=np.int0 )) if self.pool else None

        self.lock = mp.Lock()
        self.timeout = None

        self.env = env

    def evaluate_genomes(self, genomes, config):
        t0 = time.time()

        if self.pool is None:
            for genome_id, genome in genomes:
                eval_genome( self.env,genome, config )
        else:
            jobs = []
            for genome_id, genome in genomes:
                jobs.append( self.pool.apply_async( eval_genome, (self.env, genome, config) ) )

            # assign the fitness back to each genome
            for job, (ignored_genome_id, genome) in zip( jobs, genomes ):
                genome.fitness = job.get( timeout=self.timeout )
                print( 'run current evaluate_process %s ,geome id:%d,fitness %d'
                       % (os.getpid(), genome.id, genome.fitness) )
        print( "final fitness compute time {0}\n".format( time.time() - t0 ) )

class Genome():
    def __init__(self,id = None):
        self.id = id
        self.fitness = None

def run(NUM_CORES):
    g1 = Genome(1)
    g2 = Genome(2)
    g3 = Genome( 3 )

    genomes = [[1,g1],[2,g2],[3,g3]]

    exchange = Exchange( id=1, balance=100 )
    env = Env( id=1, exchange=exchange )

    pe = PooledEvaluate(NUM_CORES,env)
    pe.evaluate_genomes(genomes, config)

    print( 'run current run_process %s ,exchange balance:%d'
           % (os.getpid(), exchange.balance) )


    print('debug')

if __name__ == '__main__':
    NUM_CORES = 2
    run(NUM_CORES)
