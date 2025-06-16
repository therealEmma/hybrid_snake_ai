from enum import Enum

class DeathCause(Enum):
    WALL = "Wall Collision"
    SELF = "Self Collision"
    TIMEOUT = "Timeout" 