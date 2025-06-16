# snake_game.py
import random
import pygame
import time

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Font
pygame.font.init()
font = pygame.font.SysFont('Arial', 16)

# Grid and display settings
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400

# Set up display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game with CGP")
clock = pygame.time.Clock()

class SnakeEnv:
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (1, 0)  # Start moving right
        self.food = self._place_food()
        self.game_over = False
        self.score = 0
        return self.get_state()

    def _place_food(self):
        """Place food at a random empty position."""
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        """Return the current state as inputs for CGP."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Danger flags (wall or body)
        danger_straight = self._is_danger(head_x + self.direction[0], head_y + self.direction[1])
        danger_right = self._is_danger(head_x + self.direction[1], head_y - self.direction[0])
        danger_left = self._is_danger(head_x - self.direction[1], head_y + self.direction[0])
        
        # Food direction relative to head
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        
        # Distance to nearest body segment in each direction
        body_straight = self._distance_to_body(head_x, head_y, self.direction[0], self.direction[1])
        body_right = self._distance_to_body(head_x, head_y, self.direction[1], -self.direction[0])
        body_left = self._distance_to_body(head_x, head_y, -self.direction[1], self.direction[0])
        body_behind = self._distance_to_body(head_x, head_y, -self.direction[0], -self.direction[1])
        
        return [
            head_x, head_y,         # Snake head position
            food_x, food_y,         # Food position
            danger_straight,        # Danger ahead
            danger_right,           # Danger to the right
            danger_left,            # Danger to the left
            food_left, food_right,  # Food direction
            food_up, food_down,
            body_straight, body_right,  # Body distances
            body_left, body_behind
        ]

    def _is_danger(self, x, y):
        """Check if the position is dangerous (wall or snake body)."""
        return (x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in self.snake)

    def _distance_to_body(self, x, y, dx, dy):
        """Calculate distance to nearest body segment in given direction."""
        dist = 0
        while True:
            dist += 1
            nx, ny = x + dx * dist, y + dy * dist
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                return dist
            if (nx, ny) in self.snake[1:]:
                return dist
            if dist > max(self.width, self.height):
                return dist
        return dist

    def step(self, action):
        """Take a step based on action (0=straight, 1=right, 2=left)."""
        if self.game_over:
            return self.get_state(), self.score, True, "none"

        dx, dy = self.direction
        if action == 0:  # Straight
            new_dx, new_dy = dx, dy
        elif action == 1:  # Right turn
            new_dx, new_dy = -dy, dx
        elif action == 2:  # Left turn
            new_dx, new_dy = dy, -dx
        
        self.direction = (new_dx, new_dy)
        new_head = (self.snake[0][0] + new_dx, self.snake[0][1] + new_dy)

        # Check for collision and type
        if new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[1] >= self.height:
            self.game_over = True
            return self.get_state(), self.score, True, "wall"
        if new_head in self.snake[1:]:
            self.game_over = True
            return self.get_state(), self.score, True, "self"
        
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()

        return self.get_state(), self.score, False, "none"

    def display_info(self, text):
        """Display text information on the game screen."""
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (5, 5))

   # In snake_game.py - Update the render method:
    def render(self):
        """Render the game using Pygame."""
        try:
            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False

            screen.fill(BLACK)
            
            # Draw grid
            for x in range(0, WINDOW_WIDTH, GRID_SIZE):
                pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, WINDOW_HEIGHT))
            for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
                pygame.draw.line(screen, (50, 50, 50), (0, y), (WINDOW_WIDTH, y))
            
            # Draw snake
            for i, (x, y) in enumerate(self.snake):
                color = GREEN if i == 0 else BLUE
                pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
            # Draw food
            pygame.draw.rect(screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
            pygame.display.flip()
            clock.tick(10)
            return True
        except pygame.error:
            return False