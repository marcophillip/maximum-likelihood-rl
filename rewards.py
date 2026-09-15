from typing import List, Optional

DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


class MazeEnv:
    def __init__(self, prompt_tokens: List[str]):
        grid_start = prompt_tokens.index("GRID_START")
        grid_end = prompt_tokens.index("GRID_END")
        rows = " ".join(prompt_tokens[grid_start + 1:grid_end]).split("NEWLINE")
        self.grid = [row.split() for row in rows if row.strip()]
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
        pos = self.start
        i = -1
        for i, move in enumerate(moves):
            dr, dc = DELTAS[move]
            nxt = (pos[0] + dr, pos[1] + dc)
            cell = self._cell(*nxt)
            if cell == "WALL" or cell is None:       # hit a wall / left the grid -> episode ends, failure
                return {"solved": False, "steps": i + 1,
                        "distance": self._manhattan(pos, self.goal)}
            pos = nxt
            if pos == self.goal:
                return {"solved": True, "steps": i + 1, "distance": 0}
        return {"solved": False, "steps": i + 1, "distance": self._manhattan(pos, self.goal)}

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


def parse_moves(completion_tokens: List[str]) -> Optional[List[str]]:
    """
    Same semantics as the official MaxRL maze judge (verl/utils/reward_score/maze.py):
      * the action list is everything before the first DONE; anything after DONE is ignored
      * no DONE at all                      -> format error (None)
      * any non-move token before DONE      -> format error (None)
    """
    if "DONE" not in completion_tokens:
        return None
    moves = completion_tokens[:completion_tokens.index("DONE")]
    if any(m not in DELTAS for m in moves):
        return None
    return moves


def reward_fn(outcome: Optional[dict]) -> float:
    """
    MaxRL requires a binary terminal reward: 1.0 iff the path reaches the goal.
    Format errors (outcome is None) and failed paths both get 0.0.
    No shaping (path length, distance to goal) -- a negative or non-binary reward
    would break the mean-normalised MaxRL advantage.
    """
    return 1.0 if outcome is not None and outcome["solved"] else 0.0
