from dataclasses import dataclass

@dataclass
class GameConfig:
    screen_width: int = 800
    screen_height: int = 600
    player_radius: int = 15
    obstacle_size: int = 30
    player_x: int = 100
    step_size: int = 5
    fps: int = 60
    sensor_grid_size: int = 5
    sensor_range: int = 250
    render_grid: bool = False
    num_players: int = 1