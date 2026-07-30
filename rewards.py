
# def reward(referece_moves, generated_moves):

#     reward = (generated_moves == reference_moves) ? 1 : 0

from typing import List

DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

class MazeEnv:
    def __init__(self, prompt_tokens: List[str]):
        # TODO: slice tokens between GRID_START and GRID_END
        # TODO: split on NEWLINE -> rows -> 2D grid
        # TODO: record self.start (r, c) and self.goal (r, c)
        GRID_START_index = prompt_tokens.index("GRID_START")
        GRID_END_index = prompt_tokens.index("GRID_END")
        rows = prompt_tokens[GRID_START_index + 1: GRID_END_index]
        rows = " ".join(rows).split("NEWLINE")
        self.grid = [row.split() for row in rows]
        self.start = None
        self.goal = None

        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == "START":
                    self.start = (r, c)
                elif cell == "GOAL":
                    self.goal = (r, c)

    def _cell(self, r, c):
        if r < 0 or c < 0 or r >= len(self.grid) or c >= len(self.grid[r]):
            return None
        return self.grid[r][c]

    def simulate(self, moves: List[str]):
        """
        moves: list of moves, e.g. ["UP", "LEFT", "DOWN", ...]
        """
        pos = self.start

        illegal = 0
        i = -1                       # so "steps" is 0 when moves is empty
        for i, move in enumerate(moves):
            dr, dc = DELTAS[move]
            nxt = (pos[0] + dr, pos[1] + dc)

            cell = self._cell(*nxt)

            if cell == "WALL" or cell is None:
                illegal += 1
                break   ## stop on the first illegal move

            pos = nxt
            if pos == self.goal:
                return {"solved": True, "illegal": illegal, "steps": i + 1}
        
        return {"solved": False, 
                "illegal": illegal, 
                "steps": i + 1, 
                "distance": self._manhattan_distance(pos, self.goal)}

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def reward_fn(outcome, optimal_len):
    if outcome["solved"]:
        return optimal_len / outcome["steps"]    ### 1 if optimal else <1 if longer

    else:
        return -outcome["distance"]/28.0



