import random, time
from pprint import pprint

class VacuumCleanerAgent:
    def __init__(self, board_shape=(4,4), dirty_tile_ratio=0.4, starting_tile=None, starting_direction=None):
        
        starting_tile = starting_tile if starting_tile is not None else (random.randint(0, board_shape[0]-1), random.randint(0, board_shape[1]-1))
        starting_direction = starting_direction if starting_direction is not None else random.choice('news')
        
        self.arrows = {'n':'▲', 'e':'▶', 'w':'◀', 's':'▼'}
        self.dirty_arrows = {'n':'⇈', 'e':'⇉', 'w':'⇇', 's':'⇊'}
        self.next_direction = {'n':'e', 'e':'s', 's':'w', 'w':'n'}
        self.clear_tile = '□'
        self.traversed_tile = '◼'
        self.dirty_tile = 'x'
        
        self.init_board(board_shape, dirty_tile_ratio, starting_tile, starting_direction)
        self.cost = 0
        
    
    def init_board(self, board_shape, dirty_tile_ratio, starting_tile, starting_direction):
        self.tile = starting_tile
        self.direction = starting_direction
        self.arrow = self.arrows[self.direction]
        
        self.board_shape = board_shape
        self.board = [[None for _ in range(board_shape[1])] for _ in range(board_shape[0])]
        self.board[self.tile[0]][self.tile[1]] = self.arrow
        self.dirty_tiles_num = 0
        self.sucked_tiles_num = 0
        
        for i in range(board_shape[0]):
            for j in range(board_shape[1]):
                if self.board[i][j] == self.arrow:
                    continue
                self.board[i][j] = self.dirty_tile if random.random()<= dirty_tile_ratio else self.clear_tile
                if self.board[i][j] == self.dirty_tile:
                    self.dirty_tiles_num += 1
                
        
    def draw_board(self):
        pprint(self.board)
        print(self.tile)
        print()
        #time.sleep(0.5)
        
    
    def clone_board(self):
        cloned = [[self.board[i][j] for j in range(self.board_shape[1])] for i in range(self.board_shape[0])]
        return cloned
        
    def board_size(self):
        return self.board_shape[0] * self.board_shape[1]
        
        
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
        
        
    def move(self):
        self.board[self.tile[0]][self.tile[1]] = self.dirty_tile if self.arrow == self.dirty_arrows[self.direction] else self.traversed_tile
        
        self.tile = self.next_tile(self.tile, self.direction)
        
        curr_tile = self.board[self.tile[0]][self.tile[1]]
        self.arrow = self.dirty_arrows[self.direction] if curr_tile == self.dirty_tile else self.arrows[self.direction]
        self.board[self.tile[0]][self.tile[1]] = self.arrow
        
        self.cost += 1
        
        
    def turn_right(self):
        self.direction = self.next_direction[self.direction]
        self.arrow = self.arrows[self.direction] if self.arrow in self.arrows.values() else self.dirty_arrows[self.direction]
        self.board[self.tile[0]][self.tile[1]] = self.arrow
        
        
    def dirt_detected(self):
        return self.arrow in self.dirty_arrows.values()
    
    
    def suck(self):
        self.arrow = self.arrows[self.direction]
        self.board[self.tile[0]][self.tile[1]] = self.arrow
        self.cost += 2
        self.sucked_tiles_num += 1
    

# turning to the direction with less tiles
states = []
def solve_tttdwlt(agent: VacuumCleanerAgent):
    traversed_tiles = []
    while len(traversed_tiles) < agent.board_size():
        curr_tile = agent.tile
        traversed_tiles.append(curr_tile)
        agent.draw_board()
        states.append(agent.clone_board())
        
        if agent.dirt_detected():
            agent.suck()
            agent.draw_board()
            states.append(agent.clone_board())
            
        candidate_directions = {
            'n': curr_tile[0],
            'e': agent.board_shape[1] - 1 - curr_tile[1],
            'w': curr_tile[1],
            's': agent.board_shape[0] - 1 - curr_tile[0]
            }
        candidate_directions = {k:v for k,v in candidate_directions.items() if v != 0}
        selected_direction = None
        while len(candidate_directions)>0:
            selected_direction = min(candidate_directions, key=candidate_directions.get)
            next_tile = agent.next_tile(curr_tile, selected_direction)
            if next_tile not in traversed_tiles:
                break
            del candidate_directions[selected_direction]
        
        if selected_direction == None:
            break
        
        while agent.direction != selected_direction:
            agent.turn_right()
            agent.draw_board()
            states.append(agent.clone_board())
            
        agent.move()
    
    pprint({'Dirty Tiles Generated': agent.dirty_tiles_num, 'Sucked Tiles':agent.sucked_tiles_num, 'Total Cost':agent.cost})
    
agent = VacuumCleanerAgent()
solve_tttdwlt(agent)

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



def create_frame(matrix, cell_size, cell_margin, agent_image):
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
                case agent.clear_tile:
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                case agent.dirty_tile:
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                case agent.traversed_tile:
                    pygame.draw.rect(frame, color_dark_gray, pygame.Rect(x, y, w, h))
                    
                case '▲':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = agent_image.copy()
                    pygame.transform.rotate(img, 0)
                    frame.blit(img, (x, y))
                case '▶':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent_image.copy(), 270)
                    frame.blit(img, (x, y))
                case '◀':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent_image.copy(), 90)
                    frame.blit(img, (x, y))
                case '▼':
                    pygame.draw.rect(frame, color_white, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent_image.copy(), 180)
                    frame.blit(img, (x, y))
                case '⇈':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = agent_image.copy()
                    pygame.transform.rotate(img, 0)
                    frame.blit(img, (x, y))
                case '⇉':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent_image.copy(), 270)
                    frame.blit(img, (x, y))
                case '⇇':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent_image.copy(), 90)
                    frame.blit(img, (x, y))
                case '⇊':
                    pygame.draw.rect(frame, color_brown, pygame.Rect(x, y, w, h))
                    img = pygame.transform.rotate(agent_image.copy(), 180)
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
agent_image = pygame.image.load("agent.png")
agent_image = pygame.transform.scale(agent_image, cell_size)




# Initialize the frame counter
frame_count = 0

while frame_count < len(states):

    # Generate a random matrix
    #matrix = np.random.choice(['AC', 'D', 'C', 'AD'], size=(4,4))
    matrix = states[frame_count]
    
    # Create a frame from the matrix
    frame = create_frame(matrix, cell_size, cell_margin, agent_image)

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

