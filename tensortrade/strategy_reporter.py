class StrategyReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass
