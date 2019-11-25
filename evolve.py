from neat_stragtegy.environment import Environment

import neat
import visualize
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import time,os



def eval_genome(env,genome, config: neat.Config = None):
    pass


class PooledEvaluate(object):
    def __init__(self,timeout,NUM_CORES = 2):
        self.timeout = timeout
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.env = Environment(NUM_CORES)
        self.env_array = multiprocessing.Array( 'b', np.zeros( NUM_CORES, dtype=np.int0 )) if self.pool else None


    def evaluate_genomes(self, genomes, config):
        t0 = time.time()

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        # print("Evaluating {0} test episodes".format(len(self.test_episodes)))
        if self.pool is None:
            for genome_id, genome in genomes:
                eval_genome(genome, config)
        else:
            jobs = []
            for genome_id, genome in genomes:
                jobs.append( self.pool.apply_async(eval_genome, (self.env,genome, config) ) )

            # assign the fitness back to each genome
            for job, (ignored_genome_id, genome) in zip( jobs, genomes ):
                genome.fitness = job.get( timeout=self.timeout )

        print("final fitness compute time {0}\n".format(time.time() - t0))


def run(NUM_CORES):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(5))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledEvaluate(NUM_CORES= NUM_CORES)

    gen_best = pop.run( ec.evaluate_genomes, 5 )

if __name__ == '__main__':
    run(NUM_CORES = 1)