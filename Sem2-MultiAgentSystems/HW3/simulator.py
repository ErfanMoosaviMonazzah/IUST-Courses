import random, time
from pprint import pprint
import networkx as nx
import json
import numpy as np
import matplotlib.pyplot as plt

# %%
ARROWS_1 = { # Alice
    'clean': '↑→←↓',
    'dirty': '↟↠↞↡'
}
ARROWS_2 = { # Bob
    'clean': '^><V',
    'dirty': '⮙⮚⮘⮛'
}
SUCKING_COST = 2
MOVING_COST = 1
ROTATION_COST = 0
DEBUG_MODE = False
MAX_ITER = 32

# %%
def next_node(curr_tile, direction):
    match direction:
        case 'n':
            next_tile = (curr_tile[0]-1, curr_tile[1])
        case 'e':
            next_tile = (curr_tile[0], curr_tile[1]+1)
        case 'w':
            next_tile = (curr_tile[0], curr_tile[1]-1)
        case 's':
            next_tile = (curr_tile[0]+1, curr_tile[1])
    return next_tile

def node_valid(tile, min, max):
    return tile[0] >= min and tile[0] <= max and tile[1] >= min and tile[1] <= max

def node_visited(node, direction, visited_nodes):
    for nd in visited_nodes:
        n, d = nd
        if node[0] == n[0] and node[1] == n[1] and direction == d:
            return True
    return False

def visited_count(visited_nodes, count):
    nodes = set([node for node, direction in visited_nodes])
    return len(nodes) == count

def valid_path(ls, count):
    valids = []
    for i, visited_nodes in enumerate(ls):
        travs = []
        prev_node = curr_node = None
        for node, direction in visited_nodes:
            curr_node = node
            if curr_node != prev_node:
                if curr_node in travs:
                    break
                travs.append(curr_node)
            prev_node = node
        
        if len(travs) == count:
            valids.append(i)
    
    return valids

def path_end_at(paths, node):
    valids = []
    for i, path in enumerate(paths):
        enode = path[-1][0]
        if enode == node:
            valids.append(i)
    return valids
# %%
next_direction = {'n':'e', 'e':'s', 's':'w', 'w':'n'}
ls_travs = []

def dfs_search(node, target_node, travs=[], depth=1, border_min=0, border_max=3):
    t = travs + [node]    
    
    children = [next_node(node, w) for w in 'news' if node_valid(next_node(node, w), border_min, border_max) and node not in travs]
    
    if node == target_node:
        ls_travs.append(t)
        return depth
    
    for child in children:
        dfs_search(child, target_node, t, depth+1, border_min, border_max)
    
    return depth

# %%
class VacuumCleanerAgent():
    def __init__(self, name, starting_tile, starting_direction, arrows) -> None:
        self.arrows_clean = {k:v for k,v in zip('news', arrows['clean'])}
        self.arrows_dirty = {k:v for k,v in zip('news', arrows['dirty'])}
        self.next_direction = {'n':'e', 'e':'s', 's':'w', 'w':'n'}
        
        self.tile = starting_tile
        self.direction = starting_direction
        self.arrow = self.arrows_clean[self.direction]
        
        self.cost = 0
        self.sucked_tiles = 0
        self.name = name
        
        
class Board():
    def __init__(self, ls_agents, shape=(4,4), dirty_tile_ratio=0.4) -> None:
        self.clear_tile = '□'
        self.dirty_tile = '◼'
        self.agents = ls_agents
        
        # create board matrix
        self.shape = shape
        self.size = shape[0] * shape[1]
        self.h, self.w = shape
        self.tiles = [[self.clear_tile for _ in range(self.w)] for _ in range(self.h)]
        
        # init agents on board
        for agent in ls_agents:
            r,c = agent.tile
            self.tiles[r][c] = agent.arrow
        
        # init dirty tiles
        for i in range(self.h):
            for j in range(self.w):
                if self.tiles[i][j] == self.clear_tile:
                    self.tiles[i][j] = self.dirty_tile if random.random()<= dirty_tile_ratio else self.clear_tile
    
    def next_tile(self, curr_tile, direction):
        match direction:
            case 'n':
                next_tile = (curr_tile[0]-1, curr_tile[1])
            case 'e':
                next_tile = (curr_tile[0], curr_tile[1]+1)
            case 'w':
                next_tile = (curr_tile[0], curr_tile[1]-1)
            case 's':
                next_tile = (curr_tile[0]+1, curr_tile[1])
        return next_tile
    
    
    def dirt_detected(self, agent, err=0.1):
        res = agent.arrow in agent.arrows_dirty.values()
        if np.random.rand() <= err:
            res = not res
        return res
    
    def suck(self, agent):
        agent.cost += SUCKING_COST # increase cost value
        agent.sucked_tiles += 1
        agent.arrow = agent.arrows_clean[agent.direction] # change agent arrow to its equivalent clean arrow
        
        r, c = agent.tile # get agent position on board
        self.tiles[r][c] = agent.arrow # change agent arrow on board
        
    def move(self, agent):
        curr_tile = agent.tile
        next_tile = self.next_tile(curr_tile, agent.direction)
        
        # if the agent on the current tile has clear arrow, then curr tile would be clear after the agent left it, else if would be dirty
        r, c = curr_tile
        self.tiles[r][c] = self.clear_tile if agent.arrow in agent.arrows_clean.values() else self.dirty_tile
        
        # the agent arrow on next tile would be clear if it is clear, else agent arrow would be dirty
        r, c = next_tile
        agent.arrow = agent.arrows_clean[agent.direction] if self.tiles[r][c] == self.clear_tile else agent.arrows_dirty[agent.direction]
        
        # placing agent on next tile
        r, c = next_tile
        self.tiles[r][c] = agent.arrow
        agent.tile = next_tile
        agent.cost += MOVING_COST
    
    def turn_right(self, agent):
        r, c = agent.tile
        
        # turning agent to right
        nd = agent.next_direction[agent.direction]
        agent.arrow = agent.arrows_clean[nd] if agent.arrow in agent.arrows_clean.values() else agent.arrows_dirty[nd]
        agent.direction = nd
        
        # updating agent arrow on board
        self.tiles[r][c] = agent.arrow
        agent.cost += ROTATION_COST
        
        
    def clone_tiles(self):
        cloned_tiles = [[self.tiles[i][j] for j in range(self.w)] for i in range(self.h)]
        return cloned_tiles
    
    def print(self):
        pprint(self.tiles)
        print()

class Strategy(): 
    # stochastic turning to the direction with less tiles
    def stochastic_tttdwlt(self, board: Board):
        # store available moves per agent, each agent get enough moves to traverse the board tiles exaclty one time
        agents_available_moves = {agent.name:board.size for agent in board.agents}
        agents_traveresed_tiles = [[] for _ in range(len(board.agents))]
        states = []
        
        # while there are agents with remaining moves
        iter = 0
        while sum(agents_available_moves.values()) > (16 if DEBUG_MODE else 0) and iter < MAX_ITER:
            iter += 1
            # get a random permutation of agents
            agents = random.sample(board.agents, len(board.agents))
            states.append(board.clone_tiles())
            board.print()
            
            # move each agent for one step, if there are other agents in the middle of the path, wait that step and do nothing. 
            for agent, traversed_tiles in zip(agents, agents_traveresed_tiles):
                if DEBUG_MODE:
                    print(f'{agent.name} is choosen.')
                curr_tile = agent.tile
                traversed_tiles.append(curr_tile)
                
                if DEBUG_MODE:
                    board.print()
                    states.append(board.clone_tiles())
                
                # if the agent needs to suck, let it suck
                if board.dirt_detected(agent):
                    board.suck(agent)
                    if DEBUG_MODE:
                        board.print()
                        states.append(board.clone_tiles())
                    continue
                
                # assign a score to each direction
                candidate_directions = {
                    'n': curr_tile[0],
                    'e': board.shape[1] - 1 - curr_tile[1],
                    'w': curr_tile[1],
                    's': board.shape[0] - 1 - curr_tile[0]
                }
                # remove those directions where the agent faces a wall next to it
                candidate_directions = {k:v for k,v in candidate_directions.items() if v != 0}
                
                selected_direction = None
                while len(candidate_directions)>0:
                    # select the direction with less tiles
                    selected_direction = min(candidate_directions, key=candidate_directions.get)
                    next_tile = board.next_tile(curr_tile, selected_direction)
                    # if the next tile in selected direction is traversed before, ignore it
                    if next_tile not in traversed_tiles:
                        break
                    del candidate_directions[selected_direction]
                
                if selected_direction == None:
                    break
                
                # turn the agent until it faces the selected direction, then move
                if agent.direction != selected_direction:
                    board.turn_right(agent)
                    if DEBUG_MODE:
                        board.print()
                        states.append(board.clone_tiles())
                else:
                    # if there is no other agent on next tile, then move, else do not move in this step
                    if next_tile not in [agent.tile for agent in board.agents]:
                        board.move(agent)
                        agents_available_moves[agent.name] -= 1
                        if DEBUG_MODE:
                            board.print()
                            states.append(board.clone_tiles())
                    else:
                        if DEBUG_MODE:
                            print(f'{agent.name} waits this turn')
        return states
    
    def dfs_dual(self, board: Board, pos={}):
        # get agents
        agent_a = board.agents[0]
        agent_b = board.agents[1]
        states = []
        
        # create graph from grid and load visual positions
        graph = nx.DiGraph(nx.grid_2d_graph(4,4))
        with open('positions.json') as file:
            positions = json.load(file)
            pos = dict()
            for key in positions.keys():
                val = positions[key]
                pos[tuple(val['key'])] = np.array(val['val'])
             
        # Agent A runs dfs to find all route to agent B   
        ls_travs.clear()
        d_a = dfs_search(agent_a.tile, agent_b.tile)
        travs_a = ls_travs.copy()
        print(f'Found {len(travs_a)} valid way to reach Agent B from Agent A.')

        # Agent B does the exact opposite
        ls_travs.clear()
        d_b = dfs_search(agent_b.tile, agent_a.tile)
        travs_b = ls_travs.copy()
        print(f'Found {len(travs_b)} valid way to reach Agent A from Agent B.')
        
        # They each find a way that is the same as the other one
        same_travs = []
        for ta in travs_a:
            for tb in travs_b:
                the_same = True
                for a, b in zip(ta, reversed(tb)):
                    if a != b:
                        the_same = False
                        break
                    
                if the_same:
                    same_travs.append((len(set(ta)&set(tb)), ta, tb))    
        
        selected_trav = max(same_travs, key=lambda x: x[0])
        nodes_a = selected_trav[1]    
        nodes_b = selected_trav[2]
        
        # They now know exactly which way to traverse the grid to maximize the efficiency
        path_a, path_b = [], []
        for i in range(len(nodes_a)-1):
            path_a.append((nodes_a[i], nodes_a[i+1]))
        for i in range(len(nodes_b)-1):
            path_b.append((nodes_b[i], nodes_b[i+1]))
                
        # drawing their path
        nx.draw(graph, pos, with_labels=True)
        nx.draw_networkx_edges(nx.DiGraph(path_a), pos, edge_color='r', width=4)
        plt.savefig("Graph_A.png", format="PNG")      
              
        nx.draw(graph, pos, with_labels=True)
        nx.draw_networkx_edges(nx.DiGraph(path_b), pos, edge_color='b', width=4)
        plt.savefig("Graph_B.png", format="PNG")
        
        
        states.append(board.clone_tiles())
        i_a = i_b = 1
        while True:
            node_a = nodes_a[i_a]
            node_b = nodes_b[i_b]
            
            # A
            if board.dirt_detected(agent_a):
                board.suck(agent_a)
            else:
                nnode = next_node(agent_a.tile, agent_a.direction)
                if nnode != node_a:
                    board.turn_right(agent_a)
                else:
                    if nnode == agent_b.tile:
                        break
                    
                    board.move(agent_a)
                    i_a += 1
                    
            # B
            if board.dirt_detected(agent_b):
                board.suck(agent_b)
            else:
                nnode = next_node(agent_b.tile, agent_b.direction)
                if nnode != node_b:
                    board.turn_right(agent_b)
                else:
                    if nnode == agent_a.tile:
                        break
                    
                    board.move(agent_b)
                    i_b += 1
        

            states.append(board.clone_tiles())
            
        return states, graph, pos, path_a, path_b
              

# %%
tile1 = (random.randint(0, 3), random.randint(0, 3))
tile2 = (random.randint(0, 3), random.randint(0, 3))
print(tile1)
print(tile2)
agent1 = VacuumCleanerAgent('Alice', tile1, random.choice('news'), ARROWS_1)
agent2 = VacuumCleanerAgent('Bob',   tile2, random.choice('news'), ARROWS_2)

with open('positions.json', 'r') as file:
    positions = json.load(file)
pos = dict()
for key in positions.keys():
    val = positions[key]
    pos[tuple(val['key'])] = np.array(val['val'])

board = Board([agent1, agent2])
strategy = Strategy()

states, graph, pos, path_a, path_b = strategy.dfs_dual(board, pos)

print(f'{agent1.name} sucked {agent1.sucked_tiles} for {agent1.cost} costs')
print(f'{agent2.name} sucked {agent2.sucked_tiles} for {agent2.cost} costs')

# %%
import pygame
import numpy as np

# Define the colors and cell size
bg_color = (128, 128, 128)  # gray background
color_brown = (145, 42, 42)  
color_white = (255, 255, 255)
color_dark_gray = (64, 64, 64)
cell_size = (80, 80)
cell_margin = 5  # space between cells

# Define the window size
win_size = ((cell_size[0] + cell_margin) * 4, (cell_size[1] + cell_margin) * 4)

# Initialize Pygame
pygame.init()



def create_frame(matrix, cell_size, cell_margin, agent1_image, agent2_image):
    # Create an empty surface to hold the frame
    frame = pygame.Surface((cell_size[0]*4 + cell_margin*3, cell_size[1]*4 + cell_margin*3))

    # Fill the surface with the background color
    frame.fill(bg_color)

    # Iterate over the matrix and draw each cell onto the frame
    for i in range(4):
        for j in range(4):
            # Calculate the position of the cell
            x = (cell_size[0] + cell_margin) * j + cell_margin
            y = (cell_size[1] + cell_margin) * i + cell_margin
            w, h = cell_size
            
            match matrix[i][j]:
                case board.clear_tile:
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                case board.dirty_tile:
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                # case agent.traversed_tile:
                #     pygame.draw.rect(frame, color_dark_gray, pygame.Rect(x, y, w, h))
                
                # AGENT 1
                case '↑':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = agent1_image.copy()
                    pygame.transform.rotate(img, 0)
                    frame.blit(img, (x, y))
                case '→':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent1_image.copy(), 270)
                    frame.blit(img, (x, y))
                case '←':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent1_image.copy(), 90)
                    frame.blit(img, (x, y))
                case '↓':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent1_image.copy(), 180)
                    frame.blit(img, (x, y))
                case '↟':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = agent1_image.copy()
                    pygame.transform.rotate(img, 0)
                    frame.blit(img, (x, y))
                case '↠':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent1_image.copy(), 270)
                    frame.blit(img, (x, y))
                case '↞':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent1_image.copy(), 90)
                    frame.blit(img, (x, y))
                case '↡':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent1_image.copy(), 180)
                    frame.blit(img, (x, y))
                    
                # AGENT 2
                case '^':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = agent2_image.copy()
                    pygame.transform.rotate(img, 0)
                    frame.blit(img, (x, y))
                case '>':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent2_image.copy(), 270)
                    frame.blit(img, (x, y))
                case '<':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent2_image.copy(), 90)
                    frame.blit(img, (x, y))
                case 'V':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent2_image.copy(), 180)
                    frame.blit(img, (x, y))
                case '⮙':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = agent2_image.copy()
                    pygame.transform.rotate(img, 0)
                    frame.blit(img, (x, y))
                case '⮚':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent2_image.copy(), 270)
                    frame.blit(img, (x, y))
                case '⮘':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent2_image.copy(), 90)
                    frame.blit(img, (x, y))
                case '⮛':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent2_image.copy(), 180)
                    frame.blit(img, (x, y))

                    

    # Return the frame
    return frame




def draw_frame(frame, window):
    # Draw the frame onto the window
    window.blit(frame, (0, 0))
    
    # Update the Pygame window
    pygame.display.update()
    
    
def save_frame(frame, file_name):
    # Save the frame as a PNG image file
    pygame.image.save(frame, file_name)



# Create the Pygame window
win = pygame.display.set_mode(win_size)
win.fill(bg_color)

# Initialize the clock
clock = pygame.time.Clock()
agent1_image = pygame.image.load("agent1.png")
agent2_image = pygame.image.load("agent2.png")

with open('positions.json', 'r') as file:
    positions = json.load(file)
pos = dict()
for key in positions.keys():
    val = positions[key]
    pos[tuple(val['key'])] = np.array(val['val'])

agent1_image = pygame.transform.scale(agent1_image, cell_size)
agent2_image = pygame.transform.scale(agent2_image, cell_size)




# Initialize the frame counter
frame_count = 0

while frame_count < len(states):

    # Generate a random matrix
    #matrix = np.random.choice(['AC', 'D', 'C', 'AD'], size=(4,4))
    matrix = states[frame_count]
    
    # Create a frame from the matrix
    frame = create_frame(matrix, cell_size, cell_margin, agent1_image, agent2_image)

    # Draw the frame onto the window
    win.fill(bg_color)
    draw_frame(frame, win)

    # Save the frame as a PNG image file
    save_frame(frame, f'frame_{frame_count}.png')

    # Increment the frame counter
    frame_count += 1

    # Limit the loop to 30 FPS
    clock.tick(30)

    # Check for Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Quit the loop if the window is closed
            pygame.quit()
            exit()
    
    # Update the Pygame display
    pygame.display.update()

    # Wait for 2 seconds
    pygame.time.wait(500)

