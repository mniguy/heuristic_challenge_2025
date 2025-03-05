import traceback
from time import time
from logging import Logger

from pyquoridor.exceptions import InvalidMove, InvalidFence

from action import Action
from board import GameBoard
from .util import Performance, MEGABYTES, load_ta_agent

HARD_TIME_LIMIT = 0
HARD_MEMORY_LIMIT = 0


def execute_heuristic_search(agent, initial_state: dict, logger: Logger):
    # Build a dummy board
    board = GameBoard()
    board._initialize()

    # Load TA agent if exists
    agents = {'agent': agent}
    ta_agent = load_ta_agent(agent)
    if ta_agent is not None:
        agents['ta'] = ta_agent

    # For each agent, execute the same problem.
    results = {}
    for k, a in agents.items():
        # Initialize board and log initial memory size
        board.set_to_state(initial_state, is_initial=True)
        board.reset_memory_usage()
        board.get_current_memory_usage()

        solution = None
        failure = None

        # Start to search
        logger.info(f'Begin to search using {a.name} agent.')
        time_start = time()
        try:
            solution = a.heuristic_search(board)
            assert isinstance(solution, list), \
                'Solution should be a LIST of actions. The current outcome is not a list.'
            assert all(isinstance(s, Action) for s in solution), \
                'Solution should be a list of ACTIONs. It contains an element which is not an ACTION.'
        except:
            failure = traceback.format_exc()

        # Compute how much time passed
        time_end = time()
        time_delta = int(max(0, int(time_end - time_start)))

        # Compute how much memory used during search.
        memory_usage = int(max(0, board.get_max_memory_usage()) / MEGABYTES)

        if k == 'agent' and time_delta > HARD_TIME_LIMIT > 0:
            return Performance(
                failure=f'Time limit exceeded! {time_delta} seconds passed!',
                outcome=None,
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1 # Just give submission point
            )
        if k == 'agent' and memory_usage > HARD_MEMORY_LIMIT > 0:
            return Performance(
                failure=f'Memory limit exceeded! {memory_usage} MB used!',
                outcome=None,
                search=None,
                time=time_delta,
                memory=memory_usage,
                point=1 # Just give submission point
            )

        # Now, evaluate the solution
        is_end = False
        if solution is not None:
            try:
                board.set_to_state(initial_state, is_initial=True)  # Reset to initial state
                board.simulate_action(None, *solution)
                is_end = board.is_game_end()
            except (InvalidMove, InvalidFence):
                failure = traceback.format_exc()

        results[k] = Performance(
            failure=failure,
            outcome=len(solution) if solution is not None else None,
            search=None,
            time=time_delta,
            memory=memory_usage,
            point=1 + int(is_end)
        )

    # Give points by the stage where the agent is.
    res = results['agent']

    is_beating_ta_outcome = True
    is_beating_ta_time = False
    if 'ta' in results:
        # When TA agent exists, apply TA's result.
        is_beating_ta_outcome = ((results['ta'].outcome >= res.outcome > 0)
                                 or (results['ta'].outcome == res.outcome == 0))
        is_beating_ta_time = results['ta'].time >= res.time

    is_basic_stage = (res.failure is None) and (res.point > 1) and is_beating_ta_outcome
    is_intermediate_stage = is_basic_stage and (res.memory <= 10)
    is_advanced_stage = is_intermediate_stage and is_beating_ta_time
    # TA computation time will be measured on online system.

    return Performance(
        failure=res.failure,
        outcome=res.outcome,
        search=res.search,
        time=res.time,
        memory=res.memory,
        point=1 + int(is_basic_stage) + int(is_intermediate_stage) + int(is_advanced_stage) * 2
    )
