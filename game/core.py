import pygame
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import random
from game.config import GameConfig
    
@dataclass
class PlayerState:
    y: float
    score: float
    alive: bool
    color: Tuple[int, int, int]  

@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    height: float
    speed: float

class SurvivalGame:
    def __init__(self, config: GameConfig = GameConfig(), render: bool = True):
        self.config = config
        self.render = render
        self.reset()
        if self.config.sensor_grid_size % 2 == 0:
            self.config.sensor_grid_size += 1

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((config.screen_width, config.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
    
    def reset(self):
        # Generate random colors for each player
        colors = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) 
                  for _ in range(self.config.num_players)]
        
        self.players = [PlayerState(
            y=self.config.screen_height // 2,
            score=0,
            alive=True,
            color=colors[i]
        ) for i in range(self.config.num_players)]
        
        self.obstacles: List[Obstacle] = []
        self.frame_count = 0
        self.last_obstacle_time = 0
        self.obstacle_frequency = 20  # frames between obstacles
        
    def get_sensor_grid(self, player_y: float) -> np.ndarray:
        grid_size = self.config.sensor_grid_size
        grid = np.zeros((grid_size, grid_size))
        cell_size = self.config.sensor_range / grid_size

        half_range = self.config.sensor_range / 2
        top = player_y - half_range

        for obstacle in self.obstacles:
            dx = obstacle.x - self.config.player_x
            if 0 < dx <= self.config.sensor_range:
                # Obstacle bounds
                obstacle_left = dx
                obstacle_right = dx + obstacle.width
                obstacle_top = obstacle.y - top
                obstacle_bottom = obstacle_top + obstacle.height

                start_col = max(0, int(obstacle_left / cell_size))
                end_col = min(grid_size - 1, int(obstacle_right / cell_size))

                start_row = max(0, int(obstacle_top / cell_size))
                end_row = min(grid_size - 1, int(obstacle_bottom / cell_size))

                for row in range(start_row, end_row + 1):
                    for col in range(start_col, end_col + 1):
                        grid[row, col] = 1

        return grid
        
    def get_state(self, player_index: int, include_internals: bool) -> np.ndarray:
        """Retorna o estado atual para a rede neural de um jogador específico"""
        player = self.players[player_index]
        sensor_grid = self.get_sensor_grid(player.y).flatten()
        if include_internals:
            state = np.concatenate([
                sensor_grid,
                [player.y / self.config.screen_height],
                [(7 + min(self.frame_count*2 / 1000, 14)) / 21]
            ])
        else:
            state = sensor_grid

        return state

    def add_obstacle(self):
        """Adiciona um novo obstáculo"""
        speed = 7 + min(self.frame_count*2 / 1000, 21)  
        height = self.config.obstacle_size
        y = np.random.randint(0, self.config.screen_height - height)
        
        self.obstacles.append(Obstacle(
            x=self.config.screen_width,
            y=y,
            width=self.config.obstacle_size,
            height=height,
            speed=speed
        ))
    
    def update(self, actions: List[int]):
        """Atualiza o estado do jogo com base nas ações de todos os jogadores"""
        self.frame_count += 1
        
        for i, player in enumerate(self.players):
            if not player.alive:
                continue
            
            player.score += 0.01
            
            step_size = self.config.step_size
            action = actions[i]
            if action == 1:  # up
                player.y -= step_size
            elif action == 2:  # down
                player.y += step_size
            
            if player.y < 0 + self.config.player_radius or player.y > self.config.screen_height - self.config.player_radius:
                player.alive = False
                continue  # Skip rest of logic
        
        if self.frame_count - self.last_obstacle_time > self.obstacle_frequency:
            self.add_obstacle()
            self.last_obstacle_time = self.frame_count
            
        
        for obstacle in self.obstacles[:]:
            obstacle.x -= obstacle.speed
            
            for player in self.players:
                player_center_x = self.config.player_x + self.config.player_radius
                player_center_y = player.y

                player_left = player_center_x - self.config.player_radius
                player_right = player_center_x + self.config.player_radius
                player_top = player_center_y - self.config.player_radius
                player_bottom = player_center_y + self.config.player_radius

                if player.alive and (
                    player_right > obstacle.x and
                    player_left < obstacle.x + obstacle.width and
                    player_bottom > obstacle.y and
                    player_top < obstacle.y + obstacle.height):
                    player.alive = False
            

            if obstacle.x + obstacle.width < 0:
                self.obstacles.remove(obstacle)

        if self.frame_count % 500 == 0 and self.obstacle_frequency > 8:
            self.obstacle_frequency = self.obstacle_frequency - ((self.frame_count // 500)*2)
    
    def render_frame(self):
        if not self.render:
            return
            
        self.screen.fill((0, 0, 0))
        
        # draw players
        for player in self.players:
            if player.alive:
                pygame.draw.circle(
                    self.screen, 
                    player.color, 
                    (
                        self.config.player_x + self.config.player_radius, 
                        int(player.y)
                    ), 
                    self.config.player_radius
                )
        
        # draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (obstacle.x, obstacle.y, obstacle.width, obstacle.height)
            )
        
        # draw score
        for i, player in enumerate(self.players):
            if player.alive:
                score_text = self.font.render(f"Score: {int(player.score)}", True, (255,0,0))
                self.screen.blit(score_text, (10, 10))
                break
        
        if self.config.render_grid:
            self._render_sensor_grid()

        pygame.display.flip()
        self.clock.tick(self.config.fps)
        pygame.event.pump()
    
    def all_players_dead(self) -> bool:
        return all(not player.alive for player in self.players)

    def _render_sensor_grid(self):
        for player in self.players:
            if not player.alive:
                continue

            grid = self.get_sensor_grid(player.y)
            grid_size = self.config.sensor_grid_size
            cell_size = self.config.sensor_range / grid_size
            half_range = self.config.sensor_range / 2

            grid_top = player.y - half_range

            for row in range(grid_size):
                for col in range(grid_size):
                    x = self.config.player_x + col * cell_size
                    y = grid_top + row * cell_size

                    color = (255, 0, 0) if grid[row, col] == 1 else (100, 100, 100)
                    rect = pygame.Rect(x, y, cell_size, cell_size)
                    pygame.draw.rect(self.screen, color, rect, 1)

            center_row = grid_size // 2
            x = self.config.player_x
            y = grid_top + center_row * cell_size
            pygame.draw.rect(self.screen, (0, 255, 0), (x, y, cell_size, cell_size), 2)