import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


def run_simulation(index: int, dict_states: dict, initial_state_index: int = 1,
                   num_transitions: int = 100000, error_threshold: float = 0.01,
                   error_counter_percentage: float = 0.1,
                   calculate_matrix_exponent: bool = True) -> tuple:
    num_states: int = len(dict_states)

    error_counter0: float = error_counter_percentage * num_transitions

    list_list_transitions: list = [[]] * len(dict_states)

    list_q_rows: list = [[]] * len(dict_states)

    for state_index in dict_states.keys():
        actual_state_index: int = state_index - 1

        state: tuple = dict_states[state_index]

        rate_lambda: float = state[0]
        transitions: dict = state[1]

        list_transitions: list = [0] * num_states
        list_q_columns: list = [0] * num_states

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

    t_opt: float = 0
    n_opt: int = 0
    min_error: float = 999999999
    max_error: float = min_error * -1

    converge_counters: dict = {}

    converge: bool = False

    list_stat_dist: list = [0] * len(dict_states)

    for state_index in dict_states.keys():
        state: tuple = dict_states[state_index]

        actual_state_index: int = state_index - 1

        list_stat_dist[actual_state_index] = state[2]

    arr_stat_dist = np.array(list_stat_dist)

    print_count: int = int(num_transitions * 0.1)

    while not converge and n < num_transitions:
        step: int = n

        n += 1

        vector0 = [0] * num_states

        error_pt = [0] * num_states

        if calculate_matrix_exponent:
            tQ = t * Q

            Pt = expm(tQ)

            vector0 = Pt[0]

            error_pt = vector0 - arr_stat_dist

        tup = dict_states[current_state_index]

        rate_lambda: float = tup[0]
        stationary: float = tup[2]

        scale = 1 / rate_lambda

        wait_time = np.random.exponential(scale=scale)

        if current_state_index not in dict_states_times:
            dict_states_times[current_state_index] = ([], 0)

        tup_state: tuple = dict_states_times[current_state_index]

        list_state_times: list = tup_state[0]

        accumulated_state_time: float = tup_state[1]

        accumulated_state_time += wait_time

        curr_time = t
        t += wait_time

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

        error = math.log(abs_error)

        if not calculate_matrix_exponent:
            if error < min_error:
                min_error = error

            if error > max_error:
                max_error = error

        if abs_error > error_threshold:
            converge_counters = {}
            t_opt = t
            n_opt = n
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

        error_pt_state: float = 0

        if calculate_matrix_exponent:
            error_pt_state = error_pt[actual_state_index]

            abs_error_pt_state = abs(error_pt_state)

            if abs_error_pt_state > 0:
                error_pt_state = math.log(abs_error_pt_state)

        if calculate_matrix_exponent:
            if error_pt_state < min_error:
                min_error = error_pt_state

            if error_pt_state > max_error:
                max_error = error_pt_state

        vec0: float = vector0[actual_state_index]

        list_state_times.append((curr_time, t, empirical_distribution, vec0, error, error_pt_state))

        if step % print_count == 0:
            print(f"index: {index}, n: {step}, t: {t}, wait_time: {wait_time}, lambda: {rate_lambda}"
                f", accumulated: {accumulated_state_time}"
                f", empirical distribution: {empirical_distribution}"
                f", {current_state_index0}->{current_state_index}"
                f", {error}, {vec0}, {error_pt_state}")

    return dict_states_times, t_opt, n_opt, min_error, max_error, t


def plot_error(t_opt: float, min_error: float, max_error: float,
               dict_states: dict, dict_states_times: dict,
               plot_path: str = 'ctmc1_error.png', t_max: float = 0,
               show_figures: bool = False, plot_error_pt: bool = False):
    plt.figure()

    plt.xlim(0, t_opt)
    plt.ylim(min_error, max_error)

    color = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(dict_states_times))))

    error_index: int = 0

    if plot_error_pt:
        error_index = 5
    else:
        error_index = 4

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
            y: list = [tup[error_index] for tup in list_state_times]

            c = next(color)

            plt.plot(x, y, label=f"error {state_index}", c=c)

    plt.xlabel("t")
    plt.ylabel("log of error")

    plt.legend()

    if plot_path:
        plt.savefig(plot_path)

    if show_figures:
        plt.show()


def plot_results(t_opt: float, dict_states: dict, dict_states_times: dict,
                 plot_path: str = 'ctmc1.png', t_max: float = 0,
                 plot_pt: bool = True, show_figures: bool = False):
    plt.figure()

    plt.xlim(0, t_opt)
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
            y0: list = [tup[3] for tup in list_state_times]

            c = next(color)

            plt.plot(x, y, label=f"empirical {state_index}", c=c)

            if plot_pt:
                plt.plot(x, y0, label=f"Pt {state_index}", c="black")

            plt.hlines(y=[stationary], xmin=0, xmax=x[-1], colors=c, linestyles=['-'],
                       label=f"stationary {state_index}")

    plt.xlabel("t")
    plt.ylabel("distribution")

    plt.legend()

    if plot_path:
        plt.savefig(plot_path)

    if show_figures:
        plt.show()


def run(dict_states: dict, initial_state_index: int = 1,
        num_transitions: int = 100000, error_threshold: float = 0.01,
        error_counter_percentage: float = 0.1, num_simulations: int = 10,
        plot_path: str = 'ctmc1.png', calculate_matrix_exponent: bool = True,
        dir_name: str = None, show_figures: bool = False):
    min_t_opt: float = 999999999
    opt_dict_states_times: dict = {}

    tops: list = [0] * num_simulations

    for i in range(num_simulations):
        tup: tuple = run_simulation(i, dict_states, initial_state_index, num_transitions,
                                    error_threshold, error_counter_percentage,
                                    calculate_matrix_exponent=calculate_matrix_exponent)

        t_opt: float = tup[1]
        dict_states_times = tup[0]
        min_error: float = tup[3]
        max_error: float = tup[4]
        t_max: float = tup[5]

        tops[i] = t_opt

        if min_t_opt == 0:
            min_t_opt = t_opt
            opt_dict_states_times = dict_states_times
        else:
            if t_opt < min_t_opt:
                min_t_opt = t_opt
                opt_dict_states_times = dict_states_times

    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if dir_name is None:
        dir_name = ""

    file_path: str = os.path.join(dir_name, f"{dir_name}_hist_{plot_path}")

    plt.figure()
    plt.hist(tops)
    plt.savefig(file_path)

    if show_figures:
        plt.show()

    if isinstance(opt_dict_states_times, dict) and len(opt_dict_states_times) > 0:
        file_path = os.path.join(dir_name, f"{dir_name}_error_{plot_path}")
        plot_error(min_t_opt, min_error, max_error, dict_states, opt_dict_states_times,
                   file_path, t_max, show_figures, plot_error_pt=calculate_matrix_exponent)

        file_path = os.path.join(dir_name, f"{dir_name}_{plot_path}")
        plot_results(min_t_opt, dict_states, opt_dict_states_times, file_path, t_max,
                     calculate_matrix_exponent, show_figures)


states: dict = {
    1: (1, {2: 1}, 3 / 7),
    2: (2, {1: 0.5, 3: 0.5}, 3 / 7),
    3: (3, {2: 1}, 1 / 7)
}

run(states, 1, 10000,
    error_threshold=0.0001, num_simulations=1000,
    error_counter_percentage=0.01,
    calculate_matrix_exponent=False, dir_name="no-pt")

run(states, 1, 1000,
    error_threshold=0.0001, num_simulations=3,
    error_counter_percentage=0.1,
    calculate_matrix_exponent=True, dir_name="with-pt")
