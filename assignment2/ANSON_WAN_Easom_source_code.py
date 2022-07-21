import numpy as np

class SimulatedAnnealing:
    def __init__(self, iterations, start_point, start_temp, temp_decrement_fn, alpha, beta):
        self.iterations = iterations # Iterations per temperature.
        self.start_point = start_point
        self.start_temp = start_temp
        self.alpha = alpha
        self.beta = beta

        if temp_decrement_fn == "linear":
            self.temp_decrement_fn = self.linear_decrement_fn
        elif temp_decrement_fn == "geometric":
            self.temp_decrement_fn = self.geometric_decrement_fn
        elif temp_decrement_fn == "slowDecrease":
            self.temp_decrement_fn = self.slow_decrease_decrement_fn
        else:
            print("Invalid temperature decrement function name.")
        
    def easom_function(self, tuple):
        x_1 = tuple[0]
        x_2 = tuple[1]
        return -np.cos(x_1) * np.cos(x_2) * np.exp(-((x_1 - np.pi)**2) - ((x_2 - np.pi)**2))

    def geometric_decrement_fn(self, t):
        return t * self.alpha

    def linear_decrement_fn(self, t):
        return t - self.alpha

    def slow_decrease_decrement_fn(self, t):
        return t/(1+self.beta*t)

    def neighborhood_function(self, tuple):
        x_1 = tuple[0]
        x_2 = tuple[1]
        x_1_new = np.random.uniform(max(x_1 - 5, -100), min(x_1 + 5, 100))
        x_2_new = np.random.uniform(max(x_2 - 5, -100), min(x_2 + 5, 100))
        return (x_1_new, x_2_new)

    def cost_function(self, tuple_candidate, tuple_current):
        return self.easom_function(tuple_candidate) - self.easom_function(tuple_current)

    def run(self):
        # Steps
        # 1. Generate initial candidate solution.
        # 2. Get initial temperature.
        # 3. Select temperature reduction function.
        # 4. Loop until max niumber of iterations for the tempreature.
        #     4.1. Select a new solution s_new from the neightborhood function N(s).
        #     4.2. Calculate change in cost according to the new solution. This is delta_c.
        #          - If delta_c is less than 0 then accept the new solution since it is improving.
        #          - Else, generate a random number in the range [0,1] if x < e^(-delta_c/t) then accept new solution.
        #     4.3. Decrease the temperature according to the temperature reduction function.

        current = self.start_point # Initialize solution.
        t = self.start_temp # Initialize temperature. High temps promote exploration. Low temps restrict exploration.
        end_temp = 0.001 # Termination temperature.
        
        while (t > end_temp or self.easom_function(current) == -1):
        # for _ in range(3):
            for _ in range(self.iterations):
                # Get new candidate point.
                candidate = self.neighborhood_function(current)
                cost = self.cost_function(candidate, current)
                # Check if candidate is superior to current.
                if (cost < 0):
                    # Candidate is superior to current.
                    current = candidate
                else:
                    # Candidate is not superior. Use probablity to determine acceptance.
                    if (np.random.uniform(0,1) < np.exp(-cost/t)):
                        current = candidate

            # Decrease temperature.
            t = self.temp_decrement_fn(t)
        
        print(f"Start temp: {self.start_temp}, Start point: {self.start_point} -> Result: {current}")

def test_sa_varying_starting_points():
        print("Executing SA with 10 different randomly generated starting POINTS...")

        starting_points = []
        starting_temp = 10
        end_temp = 0.001
        alpha = 0.95
        iterations = 1000

        for _ in range(10):
            starting_points.append((np.random.uniform(-100, 100), np.random.uniform(-100, 100)))

        for starting_point in starting_points:
            SimulatedAnnealing(iterations, starting_point, starting_temp, "geometric", alpha, 0).run()

def test_sa_varying_starting_temp():
        print("Executing SA with 10 different randomly generated starting TEMPERATURES...")

        starting_temps = []
        end_temp = 0.001
        starting_point = (14,25)
        alpha = 0.95
        iterations = 1000

        for _ in range(10):
            starting_temps.append(np.random.randint(3, 20))

        for starting_temp in starting_temps:
            SimulatedAnnealing(iterations, starting_point, starting_temp, "geometric", alpha, 0).run()

def test_sa_varying_annealing_schedules():
        print("Executing SA with 9 different ANNEALING SCHEDULES...")

        starting_point = (14,25)
        starting_temp = 10
        iterations = 1000

        print("Using linear annealing schedule:")
        alphas = [0.001, 0.01, 0.1]
        for alpha in alphas:
            SimulatedAnnealing(iterations, starting_point, starting_temp, "linear", alpha, 0).run()

        print("Using geometric annealing schedule:")
        alphas = [0.95, 0.995, 0.999]
        for alpha in alphas:
            SimulatedAnnealing(iterations, starting_point, starting_temp, "geometric", alpha, 0).run()

        print("Using slow-decrease annealing schedule:")
        iterations = 1 # Slow-decrease annealing schedules should only have 1 iteration per temperature.
        betas = [0.0001, 0.001, 0.01]
        for beta in betas:
            SimulatedAnnealing(iterations, starting_point, starting_temp, "geometric", 0, beta).run()

def main():
    print("Assignment 2 Question 4 Simulated Annealing for Easom Function. Written by Anson Wan")
    test_sa_varying_starting_points()
    test_sa_varying_starting_temp()
    test_sa_varying_annealing_schedules()
   
if __name__ == "__main__":
    main()