import math
import numpy as np
import matplotlib.pyplot as plt
from control import TransferFunction, feedback, step_info, step_response, series

Kp_MIN = 2.01
Kp_MAX = 17.99

Ti_MIN = 1.06
Ti_MAX = 9.41

Td_MIN = 0.27
Td_MAX = 2.36

class RealValuedGeneticAlgorithm:
    def __init__(self, num_gens=150, pop_size=50, alpha=0.5, sigma=1, crossover_prob=0.6, mutation_prob=0.25, crossover_1=True, crossover_2=False, mutation_1=True, mutation_2=False):
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.alpha = alpha # alpha value used for crossover.
        self.sigma = sigma # sigma value used for mutation.
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Use FPS Parent Selection
        # Use Elitism survival selection strategy keeping the best 2 individuals across generations

    def perfFCN(self, K_p, T_i, T_d):
        # Python version of the given perfFCN.m MATLAB function file.
        G = K_p * TransferFunction([T_i * T_d, T_i, 1], [T_i, 0])
        F = TransferFunction(1, [1,6,11,6,0])
        sys = feedback(series(G, F), 1)
        sysinf = step_info(sys)

        # Step-response
        t = np.arange(0, 100, 0.1)
        T, y = step_response(sys, T=t)

        # Integral Squared Error (ISE)
        ISE = sum((y-1) ** 2)
        
        t_r = sysinf['RiseTime']
        t_s = sysinf['SettlingTime']
        M_p = sysinf['Overshoot']

        return ISE, t_r, t_s, M_p

    def real_value_crossover(self, parent_1, parent_2, crossover_prob, alpha):
        # Whole Arithmetic Crossover (Child Generation).
        
        child_1 = []
        child_2 = []

        for gene in range(3):
            if np.random.random() < crossover_prob:
                # Perform crossover, return resulting children.
                child_1_new_gene = self.alpha * parent_2[gene] + (1-alpha) * parent_1[gene]
                child_2_new_gene = self.alpha * parent_1[gene] + (1-alpha) * parent_2[gene]
                child_1.append(round(child_1_new_gene, 2))
                child_2.append(round(child_2_new_gene, 2))
            else:
                # Do not crossover
                child_1.append(parent_1[gene])
                child_2.append(parent_2[gene])

        return child_1, child_2

    def parent_selection(self, population, fitness_values):
        # Parent selection using Fitness Proportionate Selection (FPS).
        # Roulette Wheel.
        total = sum([fitness_values[tuple(i)] for i in population])
        pick = np.random.uniform(0, total)
        spin = 0
        for i in range(len(population)):
            spin += fitness_values[tuple(population[i])]
            if spin >= pick:
                return population[i]

    def calculate_performance_of_pop(self, population):
        fitness_values = {}
        for individual in population:
            try:
                ise, t_r, t_s, M_p = self.perfFCN(individual[0], individual[1], individual[2])
                # print(f"{ise}, {t_r}, {t_s}, {M_p}")
                fitness = 10000/(ise + t_r + t_s + M_p)
                fitness_values[tuple(individual)] = fitness
                if math.isnan(fitness):
                    fitness_values[tuple(individual)] = 0
            except: # In case the ISE is 0
                fitness_values[tuple(individual)] = 0
        return fitness_values

    def mutate_children(self, child_1, child_2):
        # Gaussian Mutation.
        
        if np.random.random() < self.mutation_prob:
            child_1 += np.random.normal(0, self.sigma, size=len(child_1))
            child_2 += np.random.normal(0, self.sigma, size=len(child_2))

            # Clip mutation of child so that it fits within the problem constraints.
            child_1[0] = np.clip(child_1[0], Kp_MIN, Kp_MAX)
            child_1[1] = np.clip(child_1[1], Ti_MIN, Ti_MAX)
            child_1[2] = np.clip(child_1[2], Td_MIN, Td_MAX)
            child_2[0] = np.clip(child_2[0], Kp_MIN, Kp_MAX)
            child_2[1] = np.clip(child_2[1], Ti_MIN, Ti_MAX)
            child_2[2] = np.clip(child_2[2], Td_MIN, Td_MAX)

            # Round to 2-decimal places.
            child_1[0] = round(child_1[0], 2)
            child_1[1] = round(child_1[1], 2)
            child_1[2] = round(child_1[2], 2)
            child_2[0] = round(child_2[0], 2)
            child_2[1] = round(child_2[1], 2)
            child_2[2] = round(child_2[2], 2)
            
        return child_1, child_2

    def plot_results():
        plt.plot(generations_x, fitness_hist)
        plt.title("Fitness vs. Number of Generations")
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.savefig("a23wan_a3_q2_result")
        plt.clf()
        
    def run_rv_ga(self):
        print(f"Now running RV-GA with params:")
        print(f"num_gens: {self.num_gens}, pop_size: {self.pop_size}, crossover_prob: {self.crossover_prob}, mutation_prob: {self.mutation_prob}")

        # Create Initial Population.
        initial_pop = []
        for i in range(self.pop_size):
            K_p = round(np.random.uniform(2.01, 17.99), 2)
            T_i = round(np.random.uniform(1.06, 9.41), 2)
            T_d = round(np.random.uniform(0.27, 2.36), 2)
            initial_pop.append([K_p, T_i, T_d])

        population = initial_pop
        best_performance_each_generation = []

        # Start GA Loop.
        for iteration in range(self.num_gens):

            fitness_values = self.calculate_performance_of_pop(population)
            next_population = [] # Holds children.
            

            # Survivor Selection.
            # Apply elitism to get the 2 best solutions in the population.
            sorted_fit_vals = sorted(fitness_values, key=fitness_values.get, reverse=True) # Returns a list of keys sorted by value.
            next_population.append(list(sorted_fit_vals[0]))
            next_population.append(list(sorted_fit_vals[1]))
            best_performance_each_generation.append(fitness_values[sorted_fit_vals[0]])
            # print(f"Generation: {iteration}, Best performance: {fitness_values[sorted_fit_vals[0]]}, Best Config: {list(sorted_fit_vals[0])}")

            # Generate next generation.
            for i in range(int(self.pop_size-2/2)): # Population-2 because 2 are from elitism

                # Parent Selection
                parent_1 = self.parent_selection(population, fitness_values)
                parent_2 = self.parent_selection(population, fitness_values)

                # Crossover (Child Generation)
                child_1, child_2 = self.real_value_crossover(parent_1, parent_2, self.crossover_prob, self.alpha)
                # print(child_1)
                # print(child_2)

                # Mutation of Children
                child_1, child_2 = self.mutate_children(child_1, child_2)

                next_population.append(child_1)
                next_population.append(child_2)
            
            # Repeat
            population = next_population
        
        # Display Results.
        fitness_values = self.calculate_performance_of_pop(initial_pop)
        sorted_fit_vals = sorted(fitness_values, key=fitness_values.get, reverse=True)
        best_individual = list(sorted_fit_vals[0])
        ISE, t_r, t_s, M_p = self.perfFCN(best_individual[0], best_individual[1], best_individual[2])
        print("Initial performance: ", fitness_values[sorted_fit_vals[0]], ", Initial ISE: ", ISE, ", Parameters: ", best_individual)

        fitness_values = self.calculate_performance_of_pop(population)
        sorted_fit_vals = sorted(fitness_values, key=fitness_values.get, reverse=True)
        best_individual = list(sorted_fit_vals[0])
        ISE, t_r, t_s, M_p = self.perfFCN(best_individual[0], best_individual[1], best_individual[2])
        print("Final performance: ", fitness_values[sorted_fit_vals[0]], ", Final ISE: ", ISE, ", Parameters: ", best_individual)

        # Plot results on graph.
        filename = "num_gens{}_num_pop{}_c_prob{}_m_prob{}.png".format(self.num_gens, self.pop_size, self.crossover_prob, self.mutation_prob)
        generations = np.arange(1, self.num_gens+1)
        plt.plot(generations, best_performance_each_generation)
        plt.title("Fitness vs. Number of Generations")
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.savefig(filename)
        plt.clf()

        return best_performance_each_generation # For graphing results.

def plot_results_together(plot_title, filename, params, num_gens, fitness_each_gen_1, fitness_each_gen_2, fitness_each_gen_3):
    # Plot results on graph.
    generations_1 = np.arange(1, num_gens[0]+1)
    generations_2 = np.arange(1, num_gens[1]+1)
    generations_3 = np.arange(1, num_gens[2]+1)
    plt.plot(generations_1, fitness_each_gen_1)
    plt.plot(generations_2, fitness_each_gen_2)
    plt.plot(generations_3, fitness_each_gen_3)
    plt.title(plot_title)
    plt.ylabel("Performance")
    plt.xlabel("Generation")
    plt.legend(params)
    plt.savefig(filename)
    plt.clf()

def test_default_params():
    return RealValuedGeneticAlgorithm().run_rv_ga()

def test_two_num_gens(default_results):
    gens_50_results = RealValuedGeneticAlgorithm(num_gens=50).run_rv_ga()
    gens_225_results = RealValuedGeneticAlgorithm(num_gens=225).run_rv_ga()

    title = "Performance vs Number of Generations: Varying Number of Generations"
    filename = "compare_varying_num_gens.png"
    params = ["50","150","225"]
    plot_results_together(title, filename, params, [50,150,225], gens_50_results, default_results, gens_225_results)

def test_two_pop_sizes(default_results):
    pop_25_res = RealValuedGeneticAlgorithm(pop_size=25).run_rv_ga()
    pop_85_res = RealValuedGeneticAlgorithm(pop_size=85).run_rv_ga()

    title = "Performance vs Number of Generations: Varying Population Size"
    filename = "compare_varying_pop_size.png"
    params = ["25","50","85"]
    plot_results_together(title, filename, params, [150,150,150], pop_25_res, default_results, pop_85_res)

def test_two_crossover_probs(default_results):
    c_prob_20_res = RealValuedGeneticAlgorithm(crossover_prob=0.2).run_rv_ga()
    c_prob_90_res = RealValuedGeneticAlgorithm(crossover_prob=0.9).run_rv_ga()

    title = "Performance vs Number of Generations: Varying Crossover Probability"
    filename = "compare_varying_crossover_probs.png"
    params = ["0.2","0.6","0.9"]
    plot_results_together(title, filename, params, [150,150,150], c_prob_20_res, default_results, c_prob_90_res)

def test_two_mutation_probs(default_results):
    m_prob_5_res = RealValuedGeneticAlgorithm(mutation_prob=0.05).run_rv_ga()
    m_prob_50_res = RealValuedGeneticAlgorithm(mutation_prob=0.5).run_rv_ga()

    title = "Performance vs Number of Generations: Varying Mutation Probability"
    filename = "compare_varying_mutation_probs.png"
    params = ["0.05","0.25","0.5"]
    plot_results_together(title, filename, params, [150,150,150], m_prob_5_res, default_results, m_prob_50_res)

def main():
    default_RVGA_fitness_each_gen = test_default_params()
    test_two_num_gens(default_RVGA_fitness_each_gen)
    test_two_pop_sizes(default_RVGA_fitness_each_gen)
    test_two_crossover_probs(default_RVGA_fitness_each_gen)
    test_two_mutation_probs(default_RVGA_fitness_each_gen)

if __name__== "__main__":  # calling the main function, where the program starts running
    main()