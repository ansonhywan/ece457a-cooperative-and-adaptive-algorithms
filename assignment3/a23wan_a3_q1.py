from queue import PriorityQueue
import random

# Class that defines a entry that goes into the priority queue.
class PriorityEntry:
    def __init__(self, state, cost, swapped_indices):
        self.state = state
        self.cost = cost
        self.swapped_indices = swapped_indices

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        return f"{self.state}, Cost: {self.cost}, Swapped: {self.swapped_indices}"

class TabuSearch:
    def __init__(self, num_iter=300, tabu_tenure=10, dynamic=False, use_aspiration_1=False, use_aspiration_2=False, smaller_neighborhood=False, frequency_based=False):

        self.num_iter = num_iter
        self.tabu_tenure = tabu_tenure
        self.sol_size = 20
        self.initial_sol = random.sample(range(self.sol_size), self.sol_size)

        self.dynamic = dynamic # Use dynamic tabu tenure size.
        self.use_aspiration_1 = use_aspiration_1
        self.use_aspiration_2 = use_aspiration_2
        self.smaller_neighborhood = smaller_neighborhood # Use smaller subset of neighborhood to get next candidate.
        self.frequency_based = frequency_based # Use frequency based tabu list.

        if frequency_based is True:
            self.frequency = {}
    
        self.flows =    [[0,0,5,0,5,2,10,3,1,5,5,5,0,0,5,4,4,0,0,1],
                        [0,0,3,10,5,1,5,1,2,4,2,5,0,10,10,3,0,5,10,5],
                        [5,3,0,2,0,5,2,4,4,5,0,0,0,5,1,0,0,5,0,0],
                        [0,10,2,0,1,0,5,2,1,0,10,2,2,0,2,1,5,2,5,5],
                        [5,5,0,1,0,5,6,5,2,5,2,0,5,1,1,1,5,2,5,1],
                        [2,1,5,0,5,0,5,2,1,6,0,0,10,0,2,0,1,0,1,5],
                        [10,5,2,5,6,5,0,0,0,0,5,10,2,2,5,1,2,1,0,10],
                        [3,1,4,2,5,2,0,0,1,1,10,10,2,0,10,2,5,2,2,10],
                        [1,2,4,1,2,1,0,1,0,2,0,3,5,5,0,5,0,0,0,2],
                        [5,4,5,0,5,6,0,1,2,0,5,5,0,5,1,0,0,5,5,2],
                        [5,2,0,10,2,0,5,10,0,5,0,5,2,5,1,10,0,2,2,5],
                        [5,5,0,2,0,0,10,10,3,5,5,0,2,10,5,0,1,1,2,5],
                        [0,0,0,2,5,10,2,2,5,0,2,2,0,2,2,1,0,0,0,5],
                        [0,10,5,0,1,0,2,0,5,5,5,10,2,0,5,5,1,5,5,0],
                        [5,10,1,2,1,2,5,10,0,1,1,5,2,5,0,3,0,5,10,10],
                        [4,3,0,1,1,0,1,2,5,0,10,0,1,5,3,0,0,0,2,0],
                        [4,0,0,5,5,1,2,5,0,0,0,1,0,1,0,0,0,5,2,0],
                        [0,5,5,2,2,0,1,2,0,5,2,1,0,5,5,0,5,0,1,1],
                        [0,10,0,5,5,1,0,2,0,5,2,2,0,5,10,2,2,1,0,6],
                        [1,5,0,5,1,5,10,10,2,2,5,5,5,0,10,0,0,1,6,0]]

        self.distances =    [[0,1,2,3,4,1,2,3,4,5,2,3,4,5,6,3,4,5,6,7],
                            [1,0,1,2,3,2,1,2,3,4,3,2,3,4,5,4,3,4,5,6],
                            [2,1,0,1,2,3,2,1,2,3,4,3,2,3,4,5,4,3,4,5],
                            [3,2,1,0,1,4,3,2,1,2,5,4,3,2,3,6,5,4,3,4],
                            [4,3,2,1,0,5,4,3,2,1,6,5,4,3,2,7,6,5,4,3],
                            [1,2,3,4,5,0,1,2,3,4,1,2,3,4,5,2,3,4,5,6],
                            [2,1,2,3,4,1,0,1,2,3,2,1,2,3,4,3,2,3,4,5],
                            [3,2,1,2,3,2,1,0,1,2,3,2,1,2,3,4,3,2,3,4],
                            [4,3,2,1,2,3,2,1,0,1,4,3,2,1,2,5,4,3,2,3],
                            [5,4,3,2,1,4,3,2,1,0,5,4,3,2,1,6,5,4,3,2],
                            [2,3,4,5,6,1,2,3,4,5,0,1,2,3,4,1,2,3,4,5],
                            [3,2,3,4,5,2,1,2,3,4,1,0,1,2,3,2,1,2,3,4],
                            [4,3,2,3,4,3,2,1,2,3,2,1,0,1,2,3,2,1,2,3],
                            [5,4,3,2,3,4,3,2,1,2,3,2,1,0,1,4,3,2,1,2],
                            [6,5,4,3,2,5,4,3,2,1,4,3,2,1,0,5,4,3,2,1],
                            [3,4,5,6,7,2,3,4,5,6,1,2,3,4,5,0,1,2,3,4],
                            [4,3,4,5,6,3,2,3,4,5,2,1,2,3,4,1,0,1,2,3],
                            [5,4,3,4,5,4,3,2,3,4,3,2,1,2,3,2,1,0,1,2],
                            [6,5,4,3,4,5,4,3,2,3,4,3,2,1,2,3,2,1,0,1],
                            [7,6,5,4,3,6,5,4,3,2,5,4,3,2,1,4,3,2,1,0]]

    def calculate_cost(self, candidate):
        cost = 0
        for i in range(self.sol_size):
            for j in range(self.sol_size):
                cost += self.distances[i][j] * self.flows[candidate[i]][candidate[j]]
        return cost

    def swap(self, candidate, index_1, index_2):
        temp = candidate[index_2]
        candidate[index_2] = candidate[index_1]
        candidate[index_1] = temp

    def generate_neighbors(self, candidate, dont_generate_all=False):
        neighbors = PriorityQueue()
        x = 0
        if dont_generate_all:
            x = int(self.sol_size / 2) # Only generate half the neighbors.
        for i in range(x, self.sol_size):
            for j in range(i + 1, self.sol_size):
                new_neighbor = candidate.copy()
                self.swap(new_neighbor, i, j)
                cost = self.calculate_cost(new_neighbor)
                if self.frequency_based is True:
                    if tuple(new_neighbor) in self.frequency:
                        cost += self.frequency[tuple(new_neighbor)]*10
                indices_swapped = [i,j]
                neighbors.put(PriorityEntry(new_neighbor, cost, indices_swapped))
        return neighbors

    def run_tabu_search(self):
        cur_iter = 0
        cur_sol = self.initial_sol
        best_sol = self.initial_sol
        best_cost = self.calculate_cost(best_sol)
        tabu_list = []

        print(f"Initial Solution: {self.initial_sol}, Cost {self.calculate_cost(self.initial_sol)}")

        while (cur_iter < self.num_iter):
            # Get neighbors. Then get the best neighbor.
            if self.smaller_neighborhood is True:
                # Generate a subset of all neighbors.
                neighbors = self.generate_neighbors(cur_sol, dont_generate_all=True)
            else:
                # Generate ALL neighbors.
                neighbors = self.generate_neighbors(cur_sol)
            candidate = neighbors.get() # Since this is a PriorityQueue, the best solution is at the front.

            if candidate.swapped_indices in tabu_list:
                # Candidate is a result of a swap that is tabu. Get next best candidate that ISN'T tabu.
                if self.use_aspiration_1 is True:
                    # Best solution so far aspiration criteria.
                    # Even if this candidate is tabu, accept it if it the the best solution so far.
                    if candidate.cost < best_cost:
                        # print("Used aspiration_1 criteria here!")
                        pass
                    else:
                        # Candidate is tabu and isn't better than the best solution so far, look for new candidate that isn't tabu.
                        new_candidate = neighbors.get()
                        while (new_candidate.swapped_indices in tabu_list):
                            new_candidate = neighbors.get()
                        # Here we know the new candidate is not a result of a tabu move. It is now the candidate.
                        candidate = new_candidate
                elif self.use_aspiration_2 is True:
                    # Best solution in the neighborhood aspiration criteria.
                    # TODO: Even if the candidate is tabu, accept it if it is the best solution in the neighborhood.
                    # Since we are using a priority queue to hold neighbors, the candidate is guaranteed better than all the neighbors.
                    # Being tabu is meaningless with this aspiration criteria, search becomes hill climbing???
                    # Don't need to do anything here, the candidate is allowed even though it is tabu.
                    # print("Used aspiration_2 criteria here!")
                    pass
                else:
                    # No aspiration criteria. If the candidate is a result of a tabu move, get the next candidate that isnt tabu.
                    new_candidate = neighbors.get()
                    while (new_candidate.swapped_indices in tabu_list):
                        new_candidate = neighbors.get()
                    # Here we know the new candidate is not a result of a tabu move. It is now the candidate.
                    candidate = new_candidate

            # Check if candidate is a better solution than the best one seen so far.
            if candidate.cost <= best_cost:
                # Candidate is superior.
                cur_sol = candidate.state
                best_sol = candidate.state
                best_cost = candidate.cost
            else:
                # Candidate is not superior.
                # Remember: In tabu search, even if the candidate does not make the solution better, we still take it to the next iteration.
                cur_sol = candidate.state

            if self.frequency_based is True:
                # Add current solution to frequency map. If it is already there, add to counter.
                if tuple(cur_sol) in self.frequency:
                    self.frequency[tuple(cur_sol)] += 1
                else:
                    self.frequency[tuple(cur_sol)] = 1
                # print(self.frequency)

            
            # Update tabu list.
            if (self.tabu_tenure == len(tabu_list)):
                # Something must be removed from tabu list as it has served the length of the tabu tenure.
                tabu_list.pop(0) # Tabu List is essentially a queue, we append to end and the oldest ones are at the front (lowest indices).
            tabu_list.append(candidate.swapped_indices)

            # Dynamic tabu list size.
            if self.dynamic is True:
                # Change tabu_tenure every 50 iterations.
                if (cur_iter % 50 == 0):
                    new_tabu_tenure = random.randint(3,20)
                    if new_tabu_tenure < self.tabu_tenure:
                        # If the new tabu tenure is shorter than the current one, must slice / remove the oldest moves in tabu list.
                        # Oldest moves are at the front of the tabu list.
                        tabu_list = tabu_list[-new_tabu_tenure:] # This slices so it's only the last X elements in the list.
                    self.tabu_tenure = new_tabu_tenure

            cur_iter += 1

        print(f"Best Solution: {best_sol}, Cost {best_cost}")

def test_tabu_with_ten_different_initial_solutions():
    for _ in range(10):
        test_basic_tabu()

def test_tabu_with_smaller_and_larger_tabu_tenure():
    # Default tabu list length is set to 10. Try 5 and 15.
    print("Tabu Tenure = 5:")
    for _ in range(10):
        TabuSearch(tabu_tenure=5).run_tabu_search()
    print("Tabu Tenure = 15:")
    for _ in range(10):
        TabuSearch(tabu_tenure=15).run_tabu_search()

def test_dynamic_tabu_list_length():
    for _ in range(10):
        TabuSearch(dynamic=True).run_tabu_search()

def test_tabu_with_aspiration_criteria():
    print("Now running tabu search with best solution so far aspiration criteria:")
    for _ in range(10):
        TabuSearch(use_aspiration_1=True).run_tabu_search()
    print("Now running tabu search with best solution in neighborhood aspiration criteria:")
    for _ in range(10):
        TabuSearch(use_aspiration_2=True).run_tabu_search()

def test_tabu_using_less_than_whole_neighborhood():
    print("Now running tabu search using less than the whole neighborhood to select the next solution:")
    for _ in range(10):
        TabuSearch(smaller_neighborhood=True).run_tabu_search()

def test_tabu_with_frequency_based_tabu_list():
    print("Now running tabu search with a frequency based tabu list:")
    for _ in range(10):
        TabuSearch(frequency_based=True).run_tabu_search()

def test_basic_tabu():
    TabuSearch().run_tabu_search()

def main():
    # Note: Optimal solution is 2570 (or 1285 if you do not double the flows)
    test_tabu_with_ten_different_initial_solutions()
    test_tabu_with_smaller_and_larger_tabu_tenure()
    test_dynamic_tabu_list_length()
    test_tabu_with_aspiration_criteria()
    test_tabu_using_less_than_whole_neighborhood()
    test_tabu_with_frequency_based_tabu_list()


if __name__== "__main__":  # calling the main function, where the program starts running
    main()

