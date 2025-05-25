import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def run(dict_states: dict, initial_state_index: int = 1,
        num_transitions: int = 1000000, error_threshold: float = 0.01, error_counter_percentage: float = 0.1):

    error_counter0: float = error_counter_percentage * num_transitions

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

    n: int = 0

    t_top: float = 0

    converge_counters: dict = {}

    converge: bool = False

    while not converge and n < num_transitions:
        n0 = n
        n += 1

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

        actual_state_index: int = current_state_index - 1

        new_actual_state_index = np.random.choice(len(P),
                                                  p=P[actual_state_index])

        current_state_index0 = current_state_index

        current_state_index = new_actual_state_index + 1

        error: float = empirical_distribution - stationary

        abs_error: float = abs(error)

        if abs_error > error_threshold:
            converge_counters = {}
            t_top = t
        else:
            if current_state_index not in converge_counters:
                converge_counters[current_state_index] = 0

            counter: int = converge_counters[current_state_index]
            converge_counters[current_state_index] = counter + 1

            counter0: int = 0

            if counter > error_counter0:
                for state_index0 in states.keys():
                    if state_index0 in converge_counters:
                        counter = converge_counters[state_index0]

                        if counter > error_counter0:
                            counter0 += 1

                if counter0 == len(states):
                    converge = True

        list_state_times.append((t0, t, empirical_distribution, error))

        print(f"n: {n0}, t: {t}, tau: {tau}, lambda: {rate_lambda}"
              f", accumulated: {accumulated_state_time}"
              f", empirical distribution: {empirical_distribution}"
              f", {current_state_index0}->{current_state_index}"
              f", {error}")

    plt.figure()

    plt.xlim(0, t)
    plt.ylim(0, 1)

    color = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(dict_states_times))))

    for state_index in dict_states_times.keys():
        stationary: float = 0

        if state_index in states:
            state: tuple = states[state_index]

            if isinstance(state, tuple) and len(state) > 0:
                stationary = state[2]

        if stationary > 0:
            state_times = dict_states_times[state_index]

            list_state_times: list = state_times[0]

            x: list = [tup[1] for tup in list_state_times]
            y: list = [tup[2] for tup in list_state_times]

            c = next(color)

            plt.plot(x, y, label=f"empirical {state_index}", c=c)
            plt.hlines(y=[stationary], xmin=0, xmax=x[-1], colors=c, linestyles=['-'],
                       label=f"stationary {state_index}")

    plt.legend()
    plt.show()

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

run(states, 1, 100000)