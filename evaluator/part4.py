import traceback
from time import time
from logging import Logger
from typing import Literal

from action import Action
from board import GameBoard
from .util import Performance, MEGABYTES, load_ta_agent, load_random_agent

HARD_TIME_LIMIT = 60
HARD_MEMORY_LIMIT = 1024


def execute_adversarial_search(agent, initial_state: dict, logger: Logger,
                               agent_type: Literal['random', 'ta'] = 'random'):
    # Set up the given problem
    board = GameBoard()
    board._initialize()

    # Load TA agent if exists
    opponents = {'random': load_random_agent(board, is_opponent=True)}
    ta_agent = load_ta_agent(board, is_opponent=True)
    if ta_agent is not None:
        opponents['ta'] = ta_agent

    # For each agent, execute the same problem.
    results = {}
    for k, opponent in opponents.items():
        # Initialize board and log initial memory size
        board.set_to_state(initial_state, is_initial=True)
        board.reset_memory_usage()
        board.get_current_memory_usage()

        failure = None
        state = initial_state

        # Set player order
        if board.get_player_id() == 'black':
            order = [opponent, agent]
        else:
            order = [agent, opponent]

        # Start to search, until the end.
        logger.info(f'Begin to search using {agent.name} agent (vs. {agent_type})')
        is_agent_win = False
        memory_usage = 0
        time_delta_max = 0
        while not board.is_game_end():
            try:
                for a in order:
                    time_start = time()  # Time limit applies for each turn
                    time_limit = time_start + HARD_TIME_LIMIT

                    action = a.adversarial_search(board, time_limit=time_limit)
                    assert isinstance(action, Action), 'Solution should be an Action.'

                    # Compute how much time passed
                    time_end = time()
                    time_delta = int(max(0, int(time_end - time_start)))
                    time_delta_max = max(time_delta_max, time_delta)

                    # Execute action
                    state = board.simulate_action(state, action)

                    # Compute how much memory used during search.
                    memory_usage = int(max(0, board.get_max_memory_usage()) / MEGABYTES)

                    if time_delta > HARD_TIME_LIMIT > 0:
                        return Performance(
                            failure=f'Time limit exceeded! {time_delta} seconds passed!',
                            outcome=float('inf'),
                            search=None,
                            time=time_delta,
                            memory=memory_usage,
                            point=1  # Just give submission point
                        )

                    if memory_usage > HARD_MEMORY_LIMIT > 0:
                        return Performance(
                            failure=f'Memory limit exceeded! {memory_usage} MB used!',
                            outcome=float('inf'),
                            search=None,
                            time=time_delta,
                            memory=memory_usage,
                            point=1  # Just give submission point
                        )

                    # Check whether the game ends
                    if board.is_game_end():
                        is_agent_win = (a == agent)
                        break
            except:
                failure = traceback.format_exc()
                break

        results[k] = Performance(
            failure=failure,
            outcome=float(is_agent_win),
            search=None,
            time=time_delta_max,
            memory=memory_usage,
            point=1  # Points will be given by average value.
        )

    # Give points by the stage where the agent is.
    res = results['random']

    is_beating_ta_outcome = False
    if 'ta' in results:
        # When TA agent exists, apply TA's result.
        is_beating_ta_outcome = results['ta'].outcome == 1.0

    is_basic_stage = (res.failure is None) and (res.outcome == 1.0)
    is_intermediate_stage = is_basic_stage and (res.time <= 5)
    is_advanced_stage = is_intermediate_stage and is_beating_ta_outcome
    # TA computation time will be measured on online system.

    return Performance(
        failure=res.failure,
        outcome=res.outcome,
        search=res.search,
        time=res.time,
        memory=res.memory,
        point=1 + int(is_basic_stage) + int(is_intermediate_stage) + int(is_advanced_stage) * 2
    )
