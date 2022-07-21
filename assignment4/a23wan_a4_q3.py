import numpy as np
import matplotlib.pyplot as plt

NUM_DIMENSIONS = 2
BOUNDARIES = {
    'MIN': -5,
    'MAX': 5
}

class Particle():
    def __init__(self, start_pos, w, c1, c2, id):
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vx = 0
        self.vy = 0
        self.p_best = start_pos
        self.fitness = None
        self.id = id

    def camelback_function(self, x, y):
        return (4-2.1*x**2+(x**4)/3)*x**2 + x*y + (-4+4*y**2)*y**2

    def fitness_function(self, x, y):
        # Negative of the function because we are trying to minimize.
        return -(self.camelback_function(x, y))

    def update_personal_best(self):
        self.fitness = self.fitness_function(self.x, self.y)

        # Update Pbest
        if self.fitness > self.fitness_function(self.p_best[0], self.p_best[1]):
            self.p_best = (self.x, self.y)

class Particle_Swarm_Optimization:
    def __init__(self, w, c1, c2, pop_size, max_iter, is_linear=False, verbose=False):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.is_linear = is_linear
        self.verbose = verbose
        if (self.is_linear):
            self.pso_type = "linear"
        else:
            self.pso_type = "simple"

    def camelback_function(self, x, y):
        return (4-2.1*x**2+(x**4)/3)*x**2 + x*y + (-4+4*y**2)*y**2

    def fitness_function(self, x, y):
        return -(self.camelback_function(x, y))

    def init_swarm(self, w, c1, c2, pop_size):
        particles = []
        for num in range(pop_size):
            # Generate starting position to be a random point within the boundaries of the search space.
            start_pos = np.random.uniform(BOUNDARIES['MIN'], BOUNDARIES['MAX'], size=NUM_DIMENSIONS)
            particles.append(Particle(start_pos, w, c1, c2, num))
        return particles

    def update_velocity(self, particle, r, g_best):
        # Calculate next velocity via the motion equation for PSO.
        particle.vx = (particle.w * particle.vx) + particle.c1 * r[0] * (particle.p_best[0] - particle.x) + particle.c2 * r[2] * (g_best[0] - particle.x)
        particle.vy = (particle.w * particle.vy) + particle.c1 * r[1] * (particle.p_best[1] - particle.y) + particle.c2 * r[3] * (g_best[1] - particle.y)

    def update_position(self, particle):
        # Calculate next position.
        particle.x = particle.x + particle.vx
        particle.y = particle.y + particle.vy

        # If the next position is out of bounds, set it to the boundary.
        particle.x = np.clip(particle.x, BOUNDARIES['MIN'], BOUNDARIES['MAX'])
        particle.y = np.clip(particle.y, BOUNDARIES['MIN'], BOUNDARIES['MAX'])

    def init_r_values(self, pop_size, is_linear):
        if is_linear:
            # Linear PSO requires that all particles have their own set of randomly generated r values.
            r = np.random.uniform(0, 1, size=(pop_size, NUM_DIMENSIONS*2)) # List that holds 50 arrays with 4 r values.
            return r
        else:
            # Simple PSO. Generate one set of r values that all particles will use.
            r = np.random.uniform(0, 1, size=NUM_DIMENSIONS*2)
            return [r] * pop_size

    def plot_results(self, average_fitness_per_iter, best_fitness_per_iter):
        title = f"{self.pso_type.capitalize()} PSO: Average and Best Fitness per Iteration"
        iterations = np.arange(1, self.max_iter+1)
        filename = "{}_PSO_w{}_c1{}_c2{}_pop_size{}_max_iter{}.png".format(self.pso_type, self.w, self.c1, self.c2, self.pop_size, self.max_iter)

        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title(title)
        plt.plot(iterations, average_fitness_per_iter)
        plt.plot(iterations, best_fitness_per_iter)
        plt.tight_layout()
        plt.legend(["Avg Fitness", "Best Fitness"])
        plt.savefig(filename)
        plt.clf()

    def run_PSO(self):
        '''
        -   Initialize the Swarm.
        -   While termination criteria is not met (cur_iter == max_iter).
            -   For each particle.
                -   Update the particle's velocity.
                -   Update the particle's position.
                -   Update the particle's personal best.
                -   Update the Gbest (We are implmenting PSO with asynchoronous update).
                End for
            End while
        '''
        print(f"Now running {self.pso_type.capitalize()} PSO with parameters:")
        print("w: {}, c1: {}, c2: {}, pop_size: {}, max_iter: {}, is_linear: {}".format(self.w, self.c1, self.c2, self.pop_size, self.max_iter, self.is_linear))

        iter = 0
        average_fitness_per_iter = [] # For graphing.
        best_fitness_per_iter = [] # For graphing.

        swarm = self.init_swarm(self.w, self.c1, self.c2, self.pop_size)
        g_best = (swarm[0].x, swarm[0].y)

        while(iter < self.max_iter):

            r = self.init_r_values(self.pop_size, self.is_linear)
            fitness_of_each_particle = []

            for particle in swarm:

                self.update_velocity(particle, r[particle.id],g_best)
                self.update_position(particle)
                particle.update_personal_best()
                fitness_of_each_particle.append(particle.fitness)

                # Update global best within the particle level for loop for asynchronous PSO.
                if (particle.fitness > self.fitness_function(g_best[0], g_best[1])):
                    g_best = (particle.x, particle.y)

            average_fitness_per_iter.append(np.mean(fitness_of_each_particle))
            best_fitness_per_iter.append(self.fitness_function(g_best[0], g_best[1]))

            if self.verbose:
                print("Iteration: {}, Global Best: {}, Fitness: {}".format(iter, g_best, self.fitness_function(g_best[0], g_best[1])))

            iter += 1

        # Result
        print("FINAL RESULT: ")
        print("Iteration: {}, Global Best: {}, Fitness: {}".format(iter, g_best, self.fitness_function(g_best[0], g_best[1])))

        # Graph the results.
        self.plot_results(average_fitness_per_iter, best_fitness_per_iter)
        
def test_PSO(w, c1, c2, pop_size, max_iter, is_linear=False, verbose=False):
    Particle_Swarm_Optimization(w, c1, c2, pop_size, max_iter, is_linear, verbose).run_PSO()

        
def main():
    max_iter = 300
    pop_size = 60
    w = 0.7
    c1 = 2
    c2 = 2

    test_PSO(w, c1, c2, pop_size, max_iter, verbose=False)
    test_PSO(w, c1, c2, pop_size, max_iter, is_linear=True, verbose=False)

if __name__ == "__main__":
    main()