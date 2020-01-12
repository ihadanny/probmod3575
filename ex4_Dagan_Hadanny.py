import pandas as pd 
import numpy as np 
import sys

from tabulate import tabulate
from collections import defaultdict

sample_txt_path1 = 'ex4-data/sample-data-1.txt'
sample_txt_path2 = 'ex4-data/sample-data-2.txt'
data_file_1a = 'ex4-data/tree-data-1a.txt'
data_file_1b = 'ex4-data/tree-data-1b.txt'
data_file_1c = 'ex4-data/tree-data-1c.txt'

edges = [(1,2), (2,3), (2,4), (1,5), (5,6), (5,7), (1,8), (8,9), (8,10)]
observed_nodes = [3, 4, 6, 7, 9, 10]
hidden_nodes = [2, 5, 8, 1]


def psi(u, v, u_val, v_val, edges):
    if u > v:
        u, v = v, u
    p = edges[(u, v)]
    if u_val == v_val:
        return 1-p
    else:
        return p

# Collect Max from ex3:
class CollectMax:
    def __init__(self, observed, edges):
        self.local_tables = {}
        self.retrieval_tables = {}
        self.marginal_tables = {}
        self.max_prob_assignment = {}
        self.collect_msgs = {}
        self.observed = observed
        self.graph = defaultdict(list)
        self.edges = edges
        for (u, v), _ in edges.items():
            self.graph[u].append(v)
            self.graph[v].append(u)


    def collect_max(self, v, u):
        children = [child for child in self.graph[v] if child != u]
        msgs = [self.collect_max(child, v) for child in children]
        
        # compute local table
        local_table = [1, 1]
        if v in self.observed:
            local_table[1-self.observed[v]]=0
        for x_v in [0, 1]:
            for msg in msgs:
                local_table[x_v] *= msg[x_v]
        
        # compute upward message 
        if u is None:
            # v is root, return the max probability 
            for x_v in [0, 1]:
                if local_table[x_v] == max(local_table):
                    self.max_prob_assignment[v] = x_v
                    break
            return max(local_table)
        else:
            msg = [0, 0]
            self.retrieval_tables[v] = [None, None]
            
            for x_u in [0, 1]:
                for x_v in [0, 1]:
                    candidate = local_table[x_v]*psi(u, v, x_u, x_v, self.edges)
                    if candidate > msg[x_u]:
                        msg[x_u] = candidate
                        self.retrieval_tables[v][x_u] = x_v
            return msg

    def distribute_max(self, u, v):
        children = [child for child in self.graph[v] if child != u]
        if u is not None:
            self.max_prob_assignment[v] = self.retrieval_tables[v][self.max_prob_assignment[u]]
        for child in children:
            self.distribute_max(v, child)

    def collect_distribute_max(self, root):
        self.local_tables.clear()
        self.retrieval_tables.clear()
        self.max_prob_assignment.clear()
        graph = defaultdict(list)
        for (u, v), p in self.edges.items():
            graph[u].append(v)
            graph[v].append(u)
        
        max_prob = self.collect_max(root, None)
        self.distribute_max(None, root)
        return self.max_prob_assignment


# Collect Sum from ex3:
class CollectSum:
    def __init__(self, observed, edges):
        self.local_tables = {}
        self.marginal_tables = {}
        self.collect_msgs = {}
        self.observed = observed
        self.graph = defaultdict(list)
        self.edges = edges
        for (u, v), p in edges.items():
            self.graph[u].append(v)
            self.graph[v].append(u)

    def collect_sum(self, v, u):
        children = [child for child in self.graph[v] if child != u]
        msgs = [self.collect_sum(child, v) for child in children]
        
        # compute local table
        local_table = [1, 1]
        if v in self.observed:
            local_table[1-self.observed[v]]=0
        for x_v in [0, 1]:
            for msg in msgs:
                local_table[x_v] *= msg[x_v]
        self.local_tables[v] = local_table
        
        # compute upward message 
        if u is None:
            # v is root, no need to prepare a message
            self.marginal_tables[v] = local_table
            return None
        else:
            msg = [0, 0]
            for x_u in [0, 1]:
                for x_v in [0, 1]:
                    msg[x_u] += local_table[x_v]*psi(u, v, x_u, x_v, self.edges)
            self.collect_msgs[(v,u)] = msg
            return msg

    def distribute_sum(self, u, v):
        children = [child for child in self.graph[v] if child != u]
        if u is not None:
            msg_uv = [0, 0]
            self.marginal_tables[v] = [0, 0]
            for x_v in [0, 1]:
                # fix message v->u to message v->u 
                for x_u in [0, 1]:
                    msg_uv[x_v] += self.marginal_tables[u][x_u]/self.collect_msgs[(v,u)][x_u]*psi(u,v,x_u,x_v, self.edges)
                # final marginal table
                self.marginal_tables[v][x_v] = self.local_tables[v][x_v]*msg_uv[x_v]
            self.collect_msgs[(u, v)] = msg_uv
        for child in children:
            self.distribute_sum(v, child)

    def normalize(self, table):
        return [t/sum(table) for t in table]

    def collect_distribute_sum(self, root):
        self.local_tables.clear()
        self.marginal_tables.clear()
        self.collect_msgs.clear()
        
        self.collect_sum(root, None)
        self.distribute_sum(None, root)
        likelihood = sum(self.marginal_tables[root])
        conditionals = {v: self.normalize(table) for v, table in
                        self.marginal_tables.items() if v not in self.observed}
        return self.local_tables, self.collect_msgs, sum(self.marginal_tables[root])


def get_table(filename, hidden_nodes, observed_nodes):
    data = np.array(pd.read_csv(filename, delimiter='\t'), dtype=int)
    count = 0
    merged_data = np.empty((data.shape[0], len(observed_nodes) + len(hidden_nodes) + 1))
    merged_data[:] = -1

    for i, observed in enumerate(observed_nodes):
        merged_data[:, observed] = data[:, i]

    return merged_data.astype(int)

def get_log_like(data, parameters):
    sum_log_likelihood = 0
    for edge, p_change in zip(edges, parameters):
        change = np.abs(data[:,edge[0]] - data[:,edge[1]])
        change_count = np.sum(change)
        p_no_change = 1 - p_change
        log_likelihood = 0 if p_change == 0 else (np.log(p_change) * change_count) 
        log_likelihood += 0 if p_no_change == 0 else (np.log(p_no_change) * (data.shape[0] - change_count)) 
        sum_log_likelihood += log_likelihood 
    return sum_log_likelihood + np.log(0.5) * data.shape[0]

def get_parameters(data):
    result = []
    
    for edge in edges:
        change = np.abs(data[:,edge[0]] - data[:,edge[1]])
        change_count = np.sum(change)
        p_change = change_count / data.shape[0]
        result.append(p_change)
    return result

def inference_from_complete_data(filename):
    data= get_table(filename, [], list(range(1, 11)))
    parameters = get_parameters(data)
    return parameters, get_log_like(data, parameters)

def make_dict_of_row(row, observed_indices):
    result = {}
    for i, item in enumerate(row):
        if item != -1 and i in observed_indices:
            result[i] = item
    return result


def maximum_probability_inference(filename, initial_setting, observed_indices):
    data = get_table(filename, hidden_nodes, observed_nodes)
    parameters = initial_setting
    history = []
    prev_likelihood = 0
    for i in range(25):
        current_edges = dict(zip(edges, parameters))
        data_likelihood = 0
        for row_index in range(data.shape[0]):
            observed = make_dict_of_row(data[row_index], observed_indices)
            collect_max = CollectMax(observed, current_edges)
            collect_sum = CollectSum(observed, current_edges)
            _, _, likelihood = collect_sum.collect_distribute_sum(1)
            data_likelihood += np.log(likelihood) + np.log(0.5)
            max_prob_assignment = collect_max.collect_distribute_max(1)
            data[row_index] = [-1] + [max_prob_assignment[i] for i in range(1, 11)]
        new_likelihood = get_log_like(data, parameters)
        history.append(parameters + [new_likelihood, data_likelihood])
        new_parameters = get_parameters(data)
        parameters = new_parameters
        if np.isclose(prev_likelihood, new_likelihood):
            break
        prev_likelihood = new_likelihood

    return history

def expectation_maximization_inference(filename, initial_setting, observed_indices):
    data = get_table(filename, hidden_nodes, observed_nodes)
    history = []
    for i in range(25):
        sum_log_likelihood = np.log(0.5) * data.shape[0]
        current_edges = dict(zip(edges, initial_setting))
        edges_sum_change = {edge: 0 for edge in edges}
        edges_sum_no_change = {edge: 0 for edge in edges}
        max_prob_data = np.zeros(data.shape)
        for row_index in range(data.shape[0]):
            observed = make_dict_of_row(data[row_index], observed_indices)
            collect_sum = CollectSum(observed, current_edges)
            collect_max = CollectMax(observed, current_edges)
            max_prob_assignment = collect_max.collect_distribute_max(1)
            max_prob_data[row_index] = [-1] + [max_prob_assignment[i] for i in range(1, 11)]
            local_tables, messages, likelihood  = collect_sum.collect_distribute_sum(1)
            sum_log_likelihood += np.log(likelihood)
            L = likelihood
            for edge in edges:
                for i in [0, 1]:
                    edges_sum_change[edge] += (local_tables[edge[1]][i] / messages[edge][i] * local_tables[edge[0]][i-1] / messages[(edge[1], edge[0])][i-1]) / L
                    edges_sum_no_change[edge] += (local_tables[edge[1]][i] / messages[edge][i] * local_tables[edge[0]][i] / messages[(edge[1], edge[0])][i]) / L
        for edge in edges:
            edges_sum_change[edge] *= current_edges[edge]
            edges_sum_no_change[edge] *= (1 - current_edges[edge])
        new_setting = []
        for edge in edges:
            new_setting.append(edges_sum_change[edge] / (edges_sum_change[edge] + edges_sum_no_change[edge]))
        most_likely = get_log_like(max_prob_data, initial_setting)
        history.append(initial_setting + [most_likely, sum_log_likelihood])
        initial_setting = new_setting
        
    return history

if __name__ == "__main__":
    try:
        option = sys.argv[1]
    except IndexError:
        print("Usage ex4.py <option>  -  options are 'C', 'M', 'E'")
        exit()

     
    if option =='C':
        probabilities, likelihood = inference_from_complete_data(sample_txt_path1)
        
        row = probabilities + [likelihood]
        headers = edges + ['log-ld']
        print(tabulate([row], headers=headers))

    if option == 'M':
        initial_setting = [0.5] * 9
        probs = maximum_probability_inference(sample_txt_path2, initial_setting, [3, 4, 6, 7, 9, 10])
        headers = edges + ['log-prob', 'log-ld']
        print(tabulate(probs, headers=headers))

    if option == 'E':
        initial_setting = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4]
        probs = expectation_maximization_inference(sample_txt_path2, initial_setting, [3, 4, 6, 7, 9, 10])
        headers = edges + ['log-prob', 'log-ld']
        print(tabulate(probs, headers=headers))