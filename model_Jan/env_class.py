import sys
from contextlib import closing
from io import StringIO
from gym import utils, Env, spaces
from gym.utils import seeding
import numpy as np


MAP = [  # used for TaxiPickupEnvSimplified and TaxiPickupEnvAdvanced
    "+---------+",
    "| : | : : |",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "| | : | : |",
    "+---------+",
]

MAP2 = [  # used for TaxiPickupEnvStandard
    "+---------+",
    "| :R| : : |",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|G| : |B: |",
    "+---------+",
]

WALLS_EAST = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
    ]
)

WALLS_WEST = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
    ]
)


class TaxiPickupEnvAdvanced(Env):
    """
    Modified version of the Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    - There are three designated pickup locations in the grid world indicated by R(ed), G(reen) and B(lue).
    - The different pickup locations have different request rates (num. requests per timestep is Poisson distributed).
    - When the episode starts, the taxi starts off at a random square and there are no passengers.
    - When a request appears at one of the locations, the taxi should drive there and pick him/her up.
    - The taxi then teleports to a random destination cell to drop off the customer.
    - Once the passenger is dropped off, the episode ends.

    Observations:
    There are 100 discrete states since there are 25 taxi positions and 3 possible requset locations.

    Passenger/request locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: B(lue)

    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger

    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """

    metadata = {"render.modes": ["human", "ansi"]}
    action_names = ["South", "North", "East", "West", "Pickup"]

    def __init__(self):
        self.desc = np.asarray(MAP, dtype="c")

        self.req_rate = req_rate = 5e-3

        self.max_episode_len = 10000
        self.num_rows = 5
        self.num_columns = 5
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1

        self.nA = 5  # number of possible actions

        self.action_space = spaces.Discrete(self.nA)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.t = 0
        self.taxi_loc = (
            int(self.np_random.random() * self.num_rows),
            int(self.np_random.random() * self.num_columns),
        )
        self.state = np.zeros((self.num_rows, self.num_columns, 4))
        self.state[self.taxi_loc[0], self.taxi_loc[1], 0] = 1
        self.state[:, :, 2] = WALLS_EAST
        self.state[:, :, 3] = WALLS_WEST
        self.lastaction = None
        self.lastreward = None
        self.total_earnings = 0
        self.teleport = None
        return self.state

    def step(self, action):
        self.t += 1
        self.lastaction = action

        self.teleport = None
        new_row, new_col = self.taxi_loc
        reward = -1  # default reward when there is no pickup/dropoff

        if action == 0:  # south
            new_row = min(self.taxi_loc[0] + 1, self.max_row)
        elif action == 1:  # north
            new_row = max(self.taxi_loc[0] - 1, 0)
        if (
            action == 2
            and self.desc[1 + self.taxi_loc[0], 2 * self.taxi_loc[1] + 2] == b":"
        ):  # east
            new_col = min(self.taxi_loc[1] + 1, self.max_col)
        elif (
            action == 3
            and self.desc[1 + self.taxi_loc[0], 2 * self.taxi_loc[1]] == b":"
        ):  # west
            new_col = max(self.taxi_loc[1] - 1, 0)
        elif action == 4:  # pickup
            if self.state[self.taxi_loc[0], self.taxi_loc[1], 1] == 0:
                reward = -10  # not passanger to pickup at this location
            else:
                # decrease req coutner and teleport to new random dropoff location
                self.state[self.taxi_loc[0], self.taxi_loc[1], 1] -= 1
                self.teleport = (self.taxi_loc[0], self.taxi_loc[1])
                new_row = int(self.np_random.random() * self.num_rows)
                new_col = int(self.np_random.random() * self.num_columns)
                reward = 20
                self.total_earnings += reward  # update total earnings

        if self.t >= self.max_episode_len:
            return (self.state, reward, True, {})  # (nextstate, reward, done, info)

        self.state[self.taxi_loc[0], self.taxi_loc[1], 0] = 0
        self.state[new_row, new_col, 0] = 1
        self.taxi_loc = (new_row, new_col)
        self.lastreward = reward

        # add new passengers according to req rate of each location
        for r in range(self.num_rows):
            for c in range(self.num_columns):
                self.state[r, c, 1] += self.np_random.poisson(self.req_rate)

        return (
            self.encode(self.state),
            reward,
            False,
            {},
        )  # (nextstate, reward, done, info)

    def encode(self, state):
        return (state > 0).astype(float)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        # state = self.decode(self.s)
        taxi_row, taxi_col = self.taxi_loc

        for r in range(self.num_rows):
            for c in range(self.num_columns):
                if self.state[r, c, 1] > 0:
                    out[1 + r][2 * c + 1] = utils.colorize(
                        str(self.state[r, c, 1])[0], "red"
                    )
                else:
                    out[1 + r][2 * c + 1] = " "

        if self.teleport != None:
            # highlight position where the taxi picked up the customer
            out[1 + self.teleport[0]][2 * self.teleport[1] + 1] = utils.colorize(
                out[1 + self.teleport[0]][2 * self.teleport[1] + 1],
                "green",
                highlight=True,
            )

            # highlight taxi position
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "magenta", highlight=True
            )
        else:
            # highlight taxi position
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                "T: {}; Total earnings: {}\n".format(self.t - 1, self.total_earnings)
            )
            outfile.write(
                "Action: {}; Reward: {}\n".format(
                    self.action_names[self.lastaction], self.lastreward
                )
            )
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
