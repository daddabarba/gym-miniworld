import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame
from ..params import DEFAULT_PARAMS

import json

class Maze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=10,
        num_cols=10,
        room_size=3,
        max_episode_steps=None,
        load_from=None,
        save_to=None,
        base_punishment = 0,
        reward_pos = None,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25
        self.load_from = load_from
        self.save_to = save_to
        self.base_punishment = base_punishment
        self.reward_pos = reward_pos

        super().__init__(
            max_episode_steps = max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall',
                    #floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        neighborhood = {}

        if self.load_from is not None:
            with open(self.load_from, 'rt') as f:
                neighborhood = dict(map(
                    lambda x : (
                        tuple(map(
                            lambda x : int(x),
                            x[0][1:-1].split(','),
                        )),
                        x[1],
                    ),
                    json.load(f).items(),
                ))

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order

            if (j,i) not in neighborhood:
                neighbors = self.rand.subset([(0,1), (0,-1), (-1,0), (1,0)], 4)
                neighborhood[(j,i)] = neighbors
            else:
                neighbors = neighborhood[(j,i)]

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        self.box = self.place_entity(Box(color='red'))

        if self.reward_pos is not None:
            self.box.pos = np.array([self.reward_pos[0],  0.        , self.reward_pos[1]])

        self.place_agent()

        if self.save_to is not None:
            print(f"Saving map to {self.save_to}")
            with open(self.save_to, 'wt') as f:
                json.dump(
                    dict(map(
                        lambda x : (
                            str(x[0]),
                            x[1],
                        ),
                        neighborhood.items()
                    )),
                    f,
                )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        reward += self.base_punishment

        return obs, reward, done, info

class MazeS2(Maze):
    def __init__(self):
        super().__init__(num_rows=2, num_cols=2)

class MazeS3(Maze):
    def __init__(self):
        super().__init__(num_rows=3, num_cols=3)

class MazeS3Fast(Maze):
    def __init__(self, forward_step=0.7, turn_step=45):

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        max_steps = 300

        super().__init__(
            num_rows=3,
            num_cols=3,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=False
        )
