
import numpy as np

def run(dict_states: dict, initial_state_index: int = 1,
        num_transitions: int = 1000000):

    list_list_transitions: list = [[]] * len(dict_states)

    list_q_rows: list = [[]] * len(dict_states)

    for state_index in dict_states.keys():
        actual_state_index: int = state_index - 1

        state: tuple = dict_states[state_index]

        rate_lambda: float = state[0]
        transitions: dict = state[1]

        list_transitions: list = [0] * len(dict_states)
        list_q_columns: list = [0] * len(dict_states)

        for trans in transitions.keys():
            actual_transition_index: int = trans - 1

            trans_prob: float = transitions[trans]

            list_transitions[actual_transition_index] = trans_prob

            list_q_columns[actual_transition_index] = trans_prob * rate_lambda

        list_list_transitions[actual_state_index] = list_transitions

        list_q_columns[actual_state_index] = rate_lambda * -1
        list_q_rows[actual_state_index] = list_q_columns

    P = np.array(list_list_transitions)

    Q = np.array(list_q_rows)

    dict_states_times: dict = {}

    current_state_index: int = initial_state_index

    t: int = 0

    for n in range(num_transitions):
        tup = dict_states[current_state_index]

        rate_lambda: float = tup[0]
        stationary: float = tup[2]

        scale = 1 / rate_lambda

        tau = np.random.exponential(scale=scale)

        if current_state_index not in dict_states_times:
            dict_states_times[current_state_index] = ([], 0)

        tup_state: tuple = dict_states_times[current_state_index]

        list_state_times: list = tup_state[0]

        accumulated_state_time: float = tup_state[1]

        accumulated_state_time += tau

        t0 = t
        t += tau

        empirical_distribution: float = accumulated_state_time / t

        dict_states_times[current_state_index] = (list_state_times,
                                                  accumulated_state_time,
                                                  empirical_distribution)

        list_state_times.append((t0, t))

        actual_state_index: int = current_state_index - 1

        new_actual_state_index = np.random.choice(len(P),
                                                  p=P[actual_state_index])

        current_state_index0 = current_state_index

        current_state_index = new_actual_state_index + 1

        error: float = empirical_distribution - stationary

        print(f"n: {n}, t: {t}, tau: {tau}, lambda: {rate_lambda}"
              f", accumulated: {accumulated_state_time}"
              f", empirical distribution: {empirical_distribution}"
              f", {current_state_index0}->{current_state_index}"
              f", {error}")

    indices: list = [0] * len(dict_states)

    finished: bool = False

    while not finished:
        min_advance: tuple = (0, 0, 0)

        found: bool = False

        for state_index in dict_states_times.keys():
            state_times = dict_states_times[state_index]

            list_state_times: list = state_times[0]

            actual_state_index: int = state_index - 1

            index: int = indices[actual_state_index]

            if index < len(list_state_times):
                found = True

                times: tuple = list_state_times[index]

                t0: float = times[1]

                min_time: float = min_advance[0]

                if min_time == 0:
                    min_advance = (t0, actual_state_index, index)
                else:
                    if t0 < min_time:
                        min_advance = (t0, actual_state_index, index)

        finished = not found

        actual_state_index: int = min_advance[1]
        index: int = min_advance[2]

        indices[actual_state_index] = index + 1

        print(min_advance)



states: dict = {
    1: (1, {2: 1}, 3/7),
    2: (2, {1: 0.5, 3: 0.5}, 3/7),
    3: (3, {2: 1}, 1/7)
}

run(states, 1, 1000)