from queue import PriorityQueue

class MazeSolver():
    def __init__(self, start=(0,0), exit=(24,24)):
        self.maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]]

        self.start = start
        self.exit = exit

    def breadth_first_search(self):
        print("Now executing BREADTH-FIRST SEARCH...")
        initial_state = {
            "position": self.start,
            "cost": 1,
            "path": [self.start]
        }
        open_queue = [initial_state]
        closed_queue = []

        while (len(open_queue) > 0):
            
            current = open_queue.pop(0)

            # Check if we are at Goal.
            if current["position"] == self.exit:
                # Current cell is the exit.
                closed_queue.append(current)
                print("Solution found!")
                print("The path is: ")
                print(current["path"])
                print(f'The cost is: {current["cost"]}')
                print(f"Number of nodes visited is: {len(closed_queue)}")
                break
            
            # Expand current state. If not already explored.
            if self.maze[current["position"][1]][current["position"][0]] == 2: # 2 means explored.
                continue
            else:
                current_pos = current["position"]
                current_x = current_pos[0]
                current_y = current_pos[1]
                # NOTE: THE ORDER OF THESE IF STATEMENTS WILL CHANGE THE RESULT OF THE SEARCH.
                if (current_x + 1 < 25) and (current_x + 1 >= 0) and (self.maze[current_y][current_x + 1] != 1): # RIGHT
                    new_path = current["path"][:]
                    new_path.append((current_x + 1, current_y))
                    open_queue.append(
                        {
                            "position": (current_x + 1, current_y),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )
                if (current_y + 1 < 25) and (current_y + 1 >= 0) and (self.maze[current_y + 1][current_x] != 1): # UP
                    new_path = current["path"][:]
                    new_path.append((current_x, current_y + 1))
                    open_queue.append(
                        {
                            "position": (current_x, current_y + 1),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )
                if (current_x - 1 < 25) and (current_x - 1 >= 0) and (self.maze[current_y][current_x - 1] != 1): # LEFT
                    new_path = current["path"][:]
                    new_path.append((current_x - 1, current_y))
                    open_queue.append(
                        {
                            "position": (current_x - 1, current_y),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )
                if (current_y - 1 < 25) and (current_y - 1 >= 0) and (self.maze[current_y - 1][current_x] != 1): # DOWN
                    new_path = current["path"][:]
                    new_path.append((current_x, current_y - 1))
                    open_queue.append(
                        {
                            "position": (current_x, current_y - 1),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )

                closed_queue.append(current) # Add the node we just expanded to the closed queue.
                self.maze[current["position"][1]][current["position"][0]] = 2 # Mark this position as explored.

    def depth_first_search(self):
        print("Now executing DEPTH-FIRST SEARCH...")
        initial_state = {
            "position": self.start,
            "cost": 1,
            "path": [self.start]
        }
        open_queue = [initial_state]
        closed_queue = []

        while (len(open_queue) > 0):

            current = open_queue.pop()

            # Check if we are at Goal.
            if current["position"] == self.exit:
                # Current cell is the exit.
                closed_queue.append(current)
                print("Solution found!")
                print("The path is: ")
                print(current["path"])
                print(f'The cost is: {current["cost"]}')
                print(f"Number of nodes visited is: {len(closed_queue)}")
                break
            
            # Expand current state. If not already explored.
            if self.maze[current["position"][1]][current["position"][0]] == 2: # 2 means explored.
                continue
            else:
                current_pos = current["position"]
                current_x = current_pos[0]
                current_y = current_pos[1]
                # NOTE: THE ORDER OF THESE IF STATEMENTS WILL CHANGE THE RESULT OF THE SEARCH.
                if (current_y - 1 < 25) and (current_y - 1 >= 0) and (self.maze[current_y - 1][current_x] != 1): # DOWN
                    new_path = current["path"][:]
                    new_path.append((current_x, current_y - 1))
                    open_queue.append(
                        {
                            "position": (current_x, current_y - 1),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )
                if (current_x - 1 < 25) and (current_x - 1 >= 0) and (self.maze[current_y][current_x - 1] != 1): # LEFT
                    new_path = current["path"][:]
                    new_path.append((current_x - 1, current_y))
                    open_queue.append(
                        {
                            "position": (current_x - 1, current_y),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )
                if (current_y + 1 < 25) and (current_y + 1 >= 0) and (self.maze[current_y + 1][current_x] != 1): # UP
                    new_path = current["path"][:]
                    new_path.append((current_x, current_y + 1))
                    open_queue.append(
                        {
                            "position": (current_x, current_y + 1),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )
                if (current_x + 1 < 25) and (current_x + 1 >= 0) and (self.maze[current_y][current_x + 1] != 1): # RIGHT
                    new_path = current["path"][:]
                    new_path.append((current_x + 1, current_y))
                    open_queue.append(
                        {
                            "position": (current_x + 1, current_y),
                            "cost": current["cost"] + 1,
                            "path": new_path
                        }
                    )

                closed_queue.append(current) # Add the node we just expanded to the closed queue.
                self.maze[current["position"][1]][current["position"][0]] = 2 # Mark this position as explored.

    # Heuristic calculation function, essentially a manhattan distance calculator given two coordinates.
    def h(self, pos_1, pos_2):
        # This is the h(n) or heursitic value of a cell.
        # We will use the Manhattan Distance as the heuristic value.
        x1,y1 = pos_1
        x2,y2 = pos_2
        return abs(x1-x2) + abs(y1-y2)

    def a_star_search(self):
        print("Now executing A-STAR SEARCH...")

        initial_state_metadata = {
            "position": self.start,
            "g": 1, # Cost
            "h": self.h(self.start, self.exit), # Manhattan Distance of self.start position and end position.
            "path": [self.start]
        }
        initial_state = PriorityEntry(initial_state_metadata['h'], initial_state_metadata)

        open_queue = PriorityQueue()
        open_queue.put(initial_state)
        closed_queue = []
        
        while not open_queue.empty():

            current = open_queue.get()

            # Check if we are at goal.
            if current.metadata["position"] == self.exit:
                # Current node is the exit.
                closed_queue.append(current.metadata)
                print("Solution found!")
                print(f"Cost is: {current.metadata['g']}")
                print(f"Path is: {current.metadata['path']}")
                print(f"Number of visited nodes is: {len(closed_queue)}")
                break

            # Expand current state. If not already explored.
            if self.maze[current.metadata["position"][1]][current.metadata["position"][0]] == 2: # 2 means explored.
                continue
            else:
                current_pos = current.metadata["position"]
                current_x = current_pos[0]
                current_y = current_pos[1]
                # NOTE: THE ORDER OF THESE IF STATEMENTS WILL CHANGE THE RESULT OF THE SEARCH.
                if (current_x + 1 < 25) and (current_x + 1 >= 0) and (self.maze[current_y][current_x + 1] != 1): # RIGHT
                    new_path = current.metadata["path"][:]
                    new_path.append((current_x + 1, current_y))
                    new_state_metadata = {
                        "position": (current.metadata["position"][0] + 1, current.metadata["position"][1]),
                        "g": current.metadata["g"] + 1,
                        "h": self.h((current.metadata["position"][0] + 1, current.metadata["position"][1]), self.exit),
                        "path": new_path
                    }
                    open_queue.put(
                        PriorityEntry(new_state_metadata['g'] + new_state_metadata['h'], new_state_metadata)
                    )
                if (current_y + 1 < 25) and (current_y + 1 >= 0) and (self.maze[current_y + 1][current_x] != 1): # UP
                    new_path = current.metadata["path"][:]
                    new_path.append((current_x, current_y + 1))
                    new_state_metadata = {
                        "position": (current.metadata["position"][0], current.metadata["position"][1] + 1),
                        "g": current.metadata["g"] + 1,
                        "h": self.h((current.metadata["position"][0], current.metadata["position"][1] + 1), self.exit),
                        "path": new_path
                    }
                    open_queue.put(
                        PriorityEntry(new_state_metadata['g'] + new_state_metadata['h'], new_state_metadata)
                    )
                if (current_x - 1 < 25) and (current_x - 1 >= 0) and (self.maze[current_y][current_x - 1] != 1): # LEFT
                    new_path = current.metadata["path"][:]
                    new_path.append((current_x - 1, current_y))
                    new_state_metadata = {
                        "position": (current.metadata["position"][0] - 1, current.metadata["position"][1]),
                        "g": current.metadata["g"] + 1,
                        "h": self.h((current.metadata["position"][0] - 1, current.metadata["position"][1]), self.exit),
                        "path": new_path
                    }
                    open_queue.put(
                        PriorityEntry(new_state_metadata['g'] + new_state_metadata['h'], new_state_metadata)
                    )
                if (current_y - 1 < 25) and (current_y - 1 >= 0) and (self.maze[current_y - 1][current_x] != 1): # DOWN
                    new_path = current.metadata["path"][:]
                    new_path.append((current_x, current_y - 1))
                    new_state_metadata = {
                        "position": (current.metadata["position"][0], current.metadata["position"][1] - 1),
                        "g": current.metadata["g"] + 1,
                        "h": self.h((current.metadata["position"][0], current.metadata["position"][1] - 1), self.exit),
                        "path": new_path
                    }
                    open_queue.put(
                        PriorityEntry(new_state_metadata['g'] + new_state_metadata['h'], new_state_metadata)
                    )
                
                closed_queue.append(current.metadata)
                self.maze[current.metadata["position"][1]][current.metadata["position"][0]] = 2 # Mark this position as explored.

# Class that defines a entry that goes into the priority queue.
class PriorityEntry:
    def __init__(self, f, metadata):
        self.f = f # f(n) = g(n) + h(n)
        self.metadata = metadata # Dictionary of metadata.

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return f"{self.f}, {self.metadata}"

def test_bfs():
    s = (2,11)
    e1 = (23,19)
    e2 = (2,21)
    MazeSolver(s, e1).breadth_first_search()
    MazeSolver(s, e2).breadth_first_search()
    MazeSolver().breadth_first_search()
    
def test_dfs():
    s = (2,11)
    e1 = (23,19)
    e2 = (2,21)
    MazeSolver(s, e1).depth_first_search()
    MazeSolver(s, e2).depth_first_search()
    MazeSolver().depth_first_search()

def test_a_star_search():
    s = (2,11)
    e1 = (23,19)
    e2 = (2,21)
    MazeSolver(s, e1).a_star_search()
    MazeSolver(s, e2).a_star_search()
    MazeSolver().a_star_search()

def main():
    print("Assignment 2 Question 2 self.maze Solver. Written by Anson Wan")
    test_bfs()
    test_dfs()
    test_a_star_search()

if __name__ == "__main__":
    main()
