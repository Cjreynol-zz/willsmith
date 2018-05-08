from enum import Enum


class GridworldDirection(Enum):
    """
    Represents the possible directions to move in Gridworld.
    """
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

    @staticmethod
    def get_offset(action):
        """
        Convert a direction to a 2D move vector.
        """
        offset = (0, 0)
        if action == GridworldDirection.UP:
            offset = (0, 1)
        elif action == GridworldDirection.RIGHT:
            offset = (1, 0)
        elif action == GridworldDirection.DOWN:
            offset = (0, -1)
        elif action == GridworldDirection.LEFT:
            offset = (-1, 0)
        else:
            raise RuntimeError("Unexpected action {}".format(action))
        return offset

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
