#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


DIM_SIZE = 2
LIMITS = {'MIN_POS': -5, 
          'MAX_POS': 5}


def six_hump_camelback(x, y):
    return (4-2.1*x**2+(x**4)/3)*x**2 + x*y + (-4+4*y**2)*y**2


def fitness_function(x, y):
    # Minimization, so fitness is negtive of function
    return -six_hump_camelback(x, y)


def init_population(pso_params, pop_size):
    population = []
    w = pso_params['w']
    c1 = pso_params['c1']
    c2 = pso_params['c2']

    for id in range(pop_size):
        init_pos = np.random.uniform(LIMITS['MIN_POS'], LIMITS['MAX_POS'], size=DIM_SIZE)
        population.append(Particle(init_pos[0], init_pos[1], w, c1, c2, id=id))
    
    return population


def get_random_numbers(pop_size, type="simple"):
    if type == "simple":
        '''
        Simple PSO - all particles use same random numbers. 
        [(r1_x, r1_y, r2_x, r2_y)] repeated pop_size times
        '''
        R = np.random.uniform(0, 1, size=DIM_SIZE*2)
        return [R] * pop_size

    elif type == "linear":
        '''
        Linear PSO - all particles use different random numbers.
        [(r1_x, r1_y, r2_x, r2_y)] unqieu for each particle
        '''
        R = np.random.uniform(0, 1, size=(pop_size, DIM_SIZE*2))
        return R
    else:
        raise NotImplementedError("Only Simple and Linear PSO implemented")


def update_velocity(p, R, Gbest):

    # Update Step Vx and Vy
    p.vx = (p.w * p.vx) + p.c1 * R[0] * (p.Pbest[0] - p.x) + p.c2 * R[2] * (Gbest[0] - p.x)
    p.vy = (p.w * p.vy) + p.c1 * R[1] * (p.Pbest[1] - p.y) + p.c2 * R[3] * (Gbest[1] - p.y)


def update_position(p):

    # Update Step x and y
    p.x = p.x + p.vx
    p.y = p.y + p.vy

    # Clip position of out of bounds
    p.x = np.clip(p.x, LIMITS['MIN_POS'], LIMITS['MAX_POS'])
    p.y = np.clip(p.y, LIMITS['MIN_POS'], LIMITS['MAX_POS'])

class Particle():

    def __init__(self, init_x, init_y, w, c1, c2, id=None):
        self.x = init_x
        self.y = init_y
        self.vx = 0
        self.vy = 0
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.id = id
        self.Pbest = (self.x, self.y)
        self.fitness = None
        self.update_fitness()

        
    def update_fitness(self):
        # minimize, so take the negative of the function
        self.fitness = fitness_function(self.x, self.y)

        # Update Pbest
        if self.fitness > fitness_function(self.Pbest[0], self.Pbest[1]):
            self.Pbest = (self.x, self.y)


def PSO(pso_params, pop_size, max_iter, type="simple", print_every=20):
    '''
    Simple PSO with Asynchronous update
    '''

    # Initialize Population
    population =  init_population(pso_params, pop_size)

    # Initialization on Global Best (x, y) as first index (shouldn't matter)
    Gbest = (population[0].x, population[0].y)

    iter = 0
    avg_fitnesses = []
    best_fitnesses = []

    while(iter < max_iter):
        
        R = get_random_numbers(pop_size, type=type)
        fitness_scores = []

        for particle in population:

            # Update each particle's velocity
            update_velocity(particle, R[particle.id], Gbest)

            # Update the particle's position
            update_position(particle)

            # Update the fitness and personal best
            particle.update_fitness()
            fitness_scores.append(particle.fitness)

            # Update the Gbest
            if particle.fitness > fitness_function(Gbest[0], Gbest[1]):
                Gbest = (particle.x, particle.y)

        avg_fitnesses.append(np.mean(fitness_scores))
        best_fitnesses.append(fitness_function(Gbest[0], Gbest[1]))

        if (iter % print_every == 0):
            print("Iteration: {}, Global Best: {}, Global Best Fitness: {}, Avg Fitness: {}".format(iter, np.round(Gbest,6), 
                    np.round(best_fitnesses[-1],4), np.round(avg_fitnesses[-1],4)))
        
        iter += 1

    # Final Performance
    print("Iteration: {}, Global Best: {}, Global Best Fitness: {}, Avg Fitness: {}".format(iter, np.round(Gbest,6), 
            np.round(best_fitnesses[-1],4), np.round(avg_fitnesses[-1],4)))

    return avg_fitnesses, best_fitnesses

def main():
    np.random.seed(457)
    pso_params = {'w': 0.792, 'c1': 1.4944, 'c2': 1.4944}
    max_iter = 250
    pop_size = 50
    print("Simple PSO:")
    simple_avg_fitnesses, simple_best_fitnesses = PSO(pso_params, pop_size, max_iter, type='simple')

    # Simple PSO Plot
    # Plot 1
    plt.plot(np.arange(1, len(simple_avg_fitnesses)+1), simple_avg_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Simple PSO: Average Fitness")
    plt.tight_layout()
    plt.savefig('simple_pso_1.png')
    plt.clf()
    # Plot 2
    plt.plot(np.arange(1, len(simple_best_fitnesses)+1), simple_best_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Simple PSO: Global Best Fitness")
    plt.tight_layout()
    plt.savefig('simple_pso_2.png')
    plt.clf()
    # Plot 3
    plt.plot(np.arange(1, len(simple_avg_fitnesses)+1), simple_avg_fitnesses)
    plt.plot(np.arange(1, len(simple_best_fitnesses)+1), simple_best_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Simple PSO for Six-Hump Camelback Function")
    plt.legend(['Average Fitness', 'Global Best Fitness'])
    plt.tight_layout()
    plt.savefig('simple_pso_3.png')
    plt.clf()

    print("Linear PSO:")
    lin_avg_fitnesses, lin_best_fitnesses = PSO(pso_params, pop_size, max_iter, type='linear')

    # Linear PSO Plot
    # Plot 1
    plt.plot(np.arange(1, len(lin_avg_fitnesses)+1), lin_avg_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Linear PSO: Average Fitness")
    plt.tight_layout()
    plt.savefig('linear_pso_1.png')
    plt.clf()
    # Plot 2
    plt.plot(np.arange(1, len(lin_best_fitnesses)+1), lin_best_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Linear PSO: Global Best Fitness")
    plt.tight_layout()
    plt.savefig('linear_pso_2.png')
    plt.clf()
    # Plot 3
    plt.plot(np.arange(1, len(lin_avg_fitnesses)+1), lin_avg_fitnesses)
    plt.plot(np.arange(1, len(lin_best_fitnesses)+1), lin_best_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Linear PSO for Six-Hump Camelback Function")
    plt.legend(['Average Fitness', 'Global Best Fitness'])
    plt.tight_layout()
    plt.savefig('linear_pso_3.png')
    plt.clf()

    # Simple PSO vs Linear PSO plot
    # Plot 1
    plt.plot(np.arange(1, len(simple_avg_fitnesses)+1), simple_avg_fitnesses)
    plt.plot(np.arange(1, len(lin_avg_fitnesses)+1), lin_avg_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Simple vs. Linear PSO: Average Fitness")
    plt.legend(['Simple PSO', 'Linear PSO'])
    plt.tight_layout()
    plt.savefig('pso_1.png')
    plt.clf()
    # Plot 2
    plt.plot(np.arange(1, len(lin_best_fitnesses)+1), lin_best_fitnesses)
    plt.plot(np.arange(1, len(simple_best_fitnesses)+1), simple_best_fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Simple vs. Linear PSO: Global Best Fitness")
    plt.legend(['Simple PSO', 'Linear PSO'])
    plt.tight_layout()
    plt.savefig('pso_2.png')
    plt.clf()


if __name__ == "__main__":
    main()
