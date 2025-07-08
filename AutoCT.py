import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import subprocess
import os
import time
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

BIT_WIDTH = 8
NUM_COLUMNS = 2 * BIT_WIDTH - 1
HIDDEN_DIM = 64
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
GRAD_CLIP = 1.0
INIT_LR = 0.001
LR_DECAY = 0.999
PRIORITY_ALPHA = 0.6
MAX_SEQ_LEN = 100  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_dc_synthesis(verilog_file, output_report_area="dc_report_area.txt",
                     output_report_timing="dc_report_timing.txt"):
    try:
        if not os.path.exists(verilog_file):
            raise FileNotFoundError(f"Verilog file {verilog_file} not found")
        dc_script = "synthesis.tcl"
        if not os.path.exists(dc_script):
            raise FileNotFoundError(f"DC script {dc_script} not found")
        result = subprocess.run(
            ["dc_shell", "-f", dc_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=150
        )
        if result.returncode != 0:
            raise RuntimeError(f"DC Synthesis failed: {result.stderr}")
        total_area = None
        if os.path.exists(output_report_area):
            with open(output_report_area, 'r') as f:
                for line in f:
                    if "Total cell area:" in line:
                        try:
                            total_area = float(line.split()[-1])
                        except (IndexError, ValueError):
                            raise ValueError("Failed to parse total cell area from report")
                        break
        total_delay = None
        if os.path.exists(output_report_timing):
            with open(output_report_timing, 'r') as f:
                for line in f:
                    if "data arrival time" in line:
                        try:
                            total_delay = float(line.split()[-1])
                        except (IndexError, ValueError):
                            raise ValueError("Failed to parse data arrival time from report")
                        break
        if total_area is not None and total_delay is not None:
            print(f"TA={total_area}, TD={total_delay}")
            return total_area, total_delay
        else:
            raise ValueError("Failed to extract area or delay from DC reports")
    except Exception as e:
        print(f"DC Synthesis failed: {e}")
        return 100.0, 5.0  

from torch_geometric.nn import GATConv

class GNNQNet(nn.Module):
    def __init__(self, input_dim, num_columns, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, edge_dim=3):
        super(GNNQNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_columns = num_columns
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_dim = edge_dim  

        
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.conv_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * num_heads, num_heads=num_heads, dropout=dropout)

        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2, hidden_dim * 2),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3 * num_columns)  
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch if hasattr(data, 'batch') else None

        
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr=edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        
        x_attn = x.view(-1, 1, x.size(-1))  
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + x_attn.squeeze(1)  

        
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_max = torch.max(x, dim=0, keepdim=True)[0]
        x_global = torch.cat([x_mean, x_max], dim=-1)  

        
        q_values = self.output(x_global)
        return q_values.view(-1, 3, self.num_columns)


class PriorityReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.pos = 0

    def add(self, transition, priority):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = transition
        self.priorities[self.pos] = (abs(priority) + 1e-5) ** PRIORITY_ALPHA
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        probs = self.priorities[:len(self.memory)]
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        return [self.memory[i] for i in indices], indices

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = (abs(p) + 1e-5) ** PRIORITY_ALPHA

# DQN Agent
class DQNAgent:
    def __init__(self, input_dim, num_columns):
        self.model = GNNQNet(input_dim, num_columns, hidden_dim=HIDDEN_DIM).to(device)
        self.target = GNNQNet(input_dim, num_columns, hidden_dim=HIDDEN_DIM).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=INIT_LR)
        self.memory = PriorityReplay(MEMORY_CAPACITY)
        self.epsilon = 1.0
        self.gamma = 0.99
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=LR_DECAY)

    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            chosen_action = random.choice(valid_actions)
            print(f"Random Action Chosen: {chosen_action}, Epsilon: {self.epsilon}")
            return chosen_action
        state = state.to(device)
        with torch.no_grad():
            q = self.model(state)[0]  # (3, num_columns)
            best_action = None
            best_q = -float('inf')
            for act in valid_actions:
                col, typ = act
                idx = {'comp53': 2, 'fa': 0, 'ha': 1}[typ]
                q_val = q[idx, col].item()
                if q_val > best_q:
                    best_q = q_val
                    best_action = act
            print(f"Greedy Action Chosen: {best_action}, Q Value: {best_q}")
        return best_action

    def remember(self, transition):
        state, action, reward, next_state, done, _ = transition
        state_gpu = state.to(device) if state is not None else None
        next_state_gpu = next_state.to(device) if next_state is not None else None
        with torch.no_grad():
            current_q = self.model(state_gpu)[
                0, 2 if action[1] == 'comp53' else (0 if action[1] == 'fa' else 1), action[0]]
            next_q = self.target(next_state_gpu).max().item() if not done else 0
            td_error = abs(reward + self.gamma * next_q - current_q.item())
        state_cpu = state  
        next_state_cpu = next_state
        transition_cpu = (state_cpu, action, reward, next_state_cpu, done, _)
        self.memory.add(transition_cpu, td_error)

    def replay(self):
        if len(self.memory.memory) < BATCH_SIZE:
            if len(self.memory.memory) == 0:
                return
            actual_batch = min(len(self.memory.memory), BATCH_SIZE)
            transitions, indices = self.memory.sample(actual_batch)
        else:
            transitions, indices = self.memory.sample(BATCH_SIZE)

        states = [t[0] for t in transitions if t[0] is not None]
        actions = [t[1] for t in transitions]
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32).to(device)
        next_states = [t[3] for t in transitions]  
        dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32).to(device)

        
        state_loader = DataLoader(states, batch_size=len(states))
        state_batch = next(iter(state_loader)).to(device)

        
        current_qs = self.model(state_batch)
        current_qs_list = []
        for s, a in zip(state_batch.to_data_list(), actions):
            q = self.model(s)[0]
            idx = 2 if a[1] == 'comp53' else (0 if a[1] == 'fa' else 1)
            current_qs_list.append(q[idx, a[0]])
        current_qs = torch.stack(current_qs_list)

        
        with torch.no_grad():
            next_qs = torch.zeros(len(transitions), dtype=torch.float32).to(device)
            valid_next_states = [ns for ns in next_states if ns is not None]
            valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
            if valid_next_states:
                next_state_loader = DataLoader(valid_next_states, batch_size=len(valid_next_states))
                next_state_batch = next(iter(next_state_loader)).to(device)
                for idx, ns in enumerate(next_state_batch.to_data_list()):
                    next_qs[valid_indices[idx]] = self.target(ns).max().item()

        targets = rewards + self.gamma * next_qs * (1 - dones)
        td_errors = (targets - current_qs).detach().cpu().numpy()
        loss = F.mse_loss(current_qs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        if hasattr(self, 'grad_norm_queue'):
            self.grad_norm_queue.append(grad_norm.item())
            if len(self.grad_norm_queue) > 100:
                self.grad_norm_queue.pop(0)
            if np.mean(self.grad_norm_queue) > 0.8 * GRAD_CLIP:
                self.lr_scheduler.step()
        else:
            self.grad_norm_queue = [grad_norm.item()]

        self.epsilon = max(0.005, self.epsilon * 0.995)

class CompressionEnv:
    def __init__(self):
        self.bit_width = BIT_WIDTH
        self.num_columns = NUM_COLUMNS
        self.pp_counts = [min(i + 1, 2 * BIT_WIDTH - 1 - i) for i in range(self.num_columns + 1)]
        self.nodes = {}  
        self.edges = defaultdict(list)  
        self.columns = [[] for _ in range(self.num_columns)]  
        self.node_to_col = {}  
        self.compression_history = []
        self.current_step = 0
        self.total_area = 0
        self.max_level = 0
        self.total_delay = 0
        self.temp_verilog_file = "temp_multiplier.v"
        
        for col, count in enumerate(self.pp_counts):
            for pos in range(count):
                signal = f'pp_col{col}_{pos}'
                self.nodes[signal] = {'type': 'pp', 'level': 0, 'signal': signal, 'origin': (col, pos)}
                self.columns[col].append(signal)
                self.node_to_col[signal] = col

    def get_state(self):
        
        node_features = []
        node_list = sorted(list(self.nodes.keys()))
        signal_to_idx = {sig: idx for idx, sig in enumerate(node_list)}
        for signal in node_list:
            node = self.nodes[signal]
            col = self.node_to_col[signal]
            feat = [
                self._type_to_float(node['type']),  
                node['level'] / 10.0,  
                len(self.columns[col]) / 10.0,  
                len(self.edges[signal]) / 5.0,  
                sum(1 for dst in self.edges[signal] if dst in self.node_to_col and self.node_to_col[dst] > col) / 5.0
                
            ]
            #print(len(self.edges[signal]) )
            node_features.append(feat)
        node_features = torch.tensor(node_features, dtype=torch.float32)

        
        edge_index = []
        edge_features = []
        for src in self.edges:
            if src in signal_to_idx:
                src_idx = signal_to_idx[src]
                src_col = self.node_to_col[src]
                src_type = self.nodes[src]['type']
                src_type_val = self._type_to_float(src_type)  
                for dst in self.edges[src]:
                    if dst in signal_to_idx:
                        dst_idx = signal_to_idx[dst]
                        dst_col = self.node_to_col[dst]
                        dst_type = self.nodes[dst]['type']
                        dst_type_val = self._type_to_float(dst_type)  
                        edge_index.append([src_idx, dst_idx])
                        
                        if 'pp' in src_type:
                            edge_type = dst_type_val * 0.5  
                        else:
                            edge_type = src_type_val * 0.3 + dst_type_val * 0.7  
                        cross_col = 1.0 if dst_col > src_col else 0.0  
                        level_diff = abs(self.nodes[src]['level'] - self.nodes[dst]['level']) / 10.0  
                        edge_features.append([edge_type, cross_col, level_diff])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.tensor(
            [[], []], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros(0, 3,
                                                                                                       dtype=torch.float32)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def _type_to_float(self, node_type):
        type_map = {
            'pp': 0.0,
            'ha_s': 0.2,
            'ha_c': 0.4,
            'fa_s': 0.6,
            'fa_c': 0.8,
            'comp53_s': 0.9,
            'comp53_c1': 0.95,
            'comp53_c2': 1.0
        }
        return type_map.get(node_type, 1.0)

    def get_valid_actions(self):
        valid_actions = []
        for col in range(self.num_columns):
            node_count = len(self.columns[col])
            if node_count >= 5:
                
                possible_types = ['comp53', 'fa', 'ha']
                chosen_type = random.choice(possible_types)
                valid_actions.append((col, chosen_type))
            elif node_count >= 3:
                
                possible_types = ['fa', 'ha']
                chosen_type = random.choice(possible_types)
                valid_actions.append((col, chosen_type))
            elif node_count >= 2:
                
                valid_actions.append((col, 'ha'))
        return valid_actions

    def execute_action(self, action, i):
        col, action_type = action
        current = self.columns[col]
        if action_type == 'comp53' and len(current) < 5:
            return -10.0, False
        if action_type == 'fa' and len(current) < 3:
            return -10.0, False
        if action_type == 'ha' and len(current) < 2:
            return -10.0, False
        if action_type == 'comp53':
            nodes = current[:5]
        elif action_type == 'fa':
            nodes = current[:3]
        else:
            nodes = current[:2]
        new_level = max(self.nodes[n]['level'] for n in nodes) + 1
        sum_sig = f"{action_type}_col{col}_L{new_level}_sum_s{self.current_step}"
        carry1_sig = f"{action_type}_col{col}_L{new_level}_carry1_s{self.current_step}"
        carry2_sig = f"{action_type}_col{col}_L{new_level}_carry2_s{self.current_step}" if action_type == 'comp53' else None

       
        self.nodes[sum_sig] = {'signal': sum_sig, 'level': new_level, 'type': f'{action_type}_s'}
        self.node_to_col[sum_sig] = col
        self.columns[col].append(sum_sig)
        for n in nodes:
            self.edges[n].append(sum_sig)
        if col + 1 < self.num_columns:
            self.nodes[carry1_sig] = {'signal': carry1_sig, 'level': new_level, 'type': f'{action_type}_c1'}
            self.node_to_col[carry1_sig] = col + 1
            self.columns[col + 1].append(carry1_sig)
            for n in nodes:
                self.edges[n].append(carry1_sig)
        if action_type == 'comp53' and col + 1 < self.num_columns:
            self.nodes[carry2_sig] = {'signal': carry2_sig, 'level': new_level, 'type': f'{action_type}_c2'}
            self.node_to_col[carry2_sig] = col + 1
            self.columns[col + 1].append(carry2_sig)
            for n in nodes:
                self.edges[n].append(carry2_sig)
        
        for n in nodes:
            self.columns[col].remove(n)
        history_entry = {
            'action_type': action_type,
            'column': col,
            'level': new_level,
            'input_signals': nodes,
            'output_sum': sum_sig,
            'output_carry1': carry1_sig,
            'output_carry2': carry2_sig if action_type == 'comp53' else None
        }
        self.compression_history.append(history_entry)
        self.current_step += 1
        self.max_level = max(self.max_level, new_level)
        return 0, all(len(col) <= 1 for col in self.columns)

    def reset(self):
        self.__init__()
        return self.get_state()


def generate_verilog(env, filename="multiplier_final.v"):
    N = BIT_WIDTH
    C = NUM_COLUMNS
    local_cols = []
    for col in range(C + 1):
        count = min(col + 1, 2 * N - 1 - col)
        nodes = [{'signal': f"pp_col{col}_{pos}"} for pos in range(count)]
        local_cols.append(nodes)
    used = set(node['signal'] for col in local_cols for node in col)

    def unique(name):
        if name is None:
            return None
        base = name
        idx = 0
        while True:
            candidate = base if idx == 0 else f"{base}_{idx}"
            if candidate not in used:
                used.add(candidate)
                return candidate
            idx += 1

    consumed = set()
    instances = []
    for rec in env.compression_history:
        col = rec['column']
        typ = rec['action_type']
        inputs = []
        for sig in rec['input_signals']:
            if sig not in consumed:
                inputs.append(sig)
                consumed.add(sig)
        for sig in inputs:
            local_cols[col] = [n for n in local_cols[col] if n['signal'] != sig]
        sum_sig = unique(rec['output_sum'])
        carry1_sig = unique(rec['output_carry1'])
        carry2_sig = unique(rec['output_carry2']) if typ == 'comp53' else None
        local_cols[col].append({'signal': sum_sig})
        if carry1_sig and col + 1 < C:
            local_cols[col + 1].append({'signal': carry1_sig})
        if carry2_sig and col + 1 < C:
            local_cols[col + 1].append({'signal': carry2_sig})
        instances.append((typ, inputs, sum_sig, carry1_sig, carry2_sig))
    final_vec1, final_vec2 = [], []
    for col_nodes in local_cols:
        uniq = []
        for node in col_nodes:
            sig = node['signal']
            if sig not in uniq:
                uniq.append(sig)
        final_vec1.append(uniq[0] if len(uniq) > 0 else "1'b0")
        final_vec2.append(uniq[1] if len(uniq) > 1 else "1'b0")
    producer = {}
    for idx, (_, _, sum_sig, carry1_sig, carry2_sig) in enumerate(instances):
        producer[sum_sig] = idx
        if carry1_sig:
            producer[carry1_sig] = idx
        if carry2_sig:
            producer[carry2_sig] = idx
    active = set(final_vec1 + final_vec2)
    queue = list(active)
    while queue:
        sig = queue.pop()
        if sig in producer:
            idx = producer[sig]
            typ, inputs, _, _, _ = instances[idx]
            for inp in inputs:
                if inp not in active:
                    active.add(inp)
                    queue.append(inp)
    instances = [
        (typ, inputs, sum_sig, carry1_sig, carry2_sig)
        for (typ, inputs, sum_sig, carry1_sig, carry2_sig) in instances
        if sum_sig in active or (carry1_sig and carry1_sig in active) or (carry2_sig and carry2_sig in active)
    ]
    decl = set()
    for col_nodes in local_cols:
        for node in col_nodes:
            if node['signal'].startswith('pp_'):
                decl.add(node['signal'])
    for typ, inputs, sum_sig, carry1_sig, carry2_sig in instances:
        decl.update(inputs)
        decl.add(sum_sig)
        if carry1_sig:
            decl.add(carry1_sig)
        if carry2_sig:
            decl.add(carry2_sig)
    decl.update(final_vec1 + final_vec2)
    with open(filename, 'w') as f:
        f.write("// --- primitive modules ---\n")
        f.write("module half_adder(input a, input b, output sum, output cout);\n")
        f.write("  assign sum = a ^ b;\n")
        f.write("  assign cout = a & b;\n")
        f.write("endmodule\n\n")
        f.write("module full_adder(input a, input b, input cin, output sum, output cout);\n")
        f.write("  assign sum  = a ^ b ^ cin;\n")
        f.write("  assign cout = (a & b) | (a & cin) | (b & cin);\n")
        f.write("endmodule\n\n")
        f.write(
            "module compressor_5_3(input a, input b, input c, input d, input e, output sum, output carry1, output carry2);\n")
        f.write("  wire t1, t2, t3;\n")
        f.write("  assign t1 = a ^ b ^ c;\n")
        f.write("  assign t2 = d ^ e;\n")
        f.write("  assign sum = t1 ^ t2;\n")
        f.write("  assign carry1 = (a & b) | (a & c) | (b & c);\n")
        f.write("  assign carry2 = (d & e) | (d & t1) | (e & t1);\n")
        f.write("endmodule\n\n")
        f.write(f"module multiplier(\n")
        f.write(f"  input  [{N - 1}:0] a,\n")
        f.write(f"  input  [{N - 1}:0] b,\n")
        f.write(f"  output [{2 * N - 1}:0] product\n")
        f.write(");\n\n")
        for sig in sorted(decl):
            if sig != '1\'b0':
                f.write(f"  wire {sig};\n")
        f.write(f"  wire [{C}:0] vec1, vec2;\n\n")
        f.write("  // partial-product assigns\n")
        for col in range(C):
            count = min(col + 1, 2 * N - 1 - col)
            offset = max(0, col - (N - 1))
            for pos in range(count):
                pp = f"pp_col{col}_{pos}"
                if pp in decl:
                    i = offset + pos
                    j = col - i
                    f.write(f"  assign {pp} = a[{i}] & b[{j}];\n")
        f.write("\n")
        f.write("  // CSA instantiation\n")
        for idx, (typ, inputs, sum_sig, carry1_sig, carry2_sig) in enumerate(instances):
            uniq_ins = []
            for s in inputs:
                if s not in uniq_ins:
                    uniq_ins.append(s)
            if typ == 'comp53' and len(uniq_ins) == 5:
                f.write(
                    f"  compressor_5_3 comp53_{idx}(.a({uniq_ins[0]}), "
                    f".b({uniq_ins[1]}), .c({uniq_ins[2]}), .d({uniq_ins[3]}), .e({uniq_ins[4]}), "
                    f".sum({sum_sig}), .carry1({carry1_sig}), .carry2({carry2_sig}));\n"
                )
            elif typ == 'fa' and len(uniq_ins) == 3:
                f.write(
                    f"  full_adder fa_{idx}(.a({uniq_ins[0]}), "
                    f".b({uniq_ins[1]}), .cin({uniq_ins[2]}), "
                    f".sum({sum_sig}), .cout({carry1_sig}));\n"
                )
            else:
                a_sig = uniq_ins[0] if uniq_ins else "1'b0"
                b_sig = uniq_ins[1] if len(uniq_ins) > 1 else "1'b0"
                f.write(
                    f"  half_adder ha_{idx}(.a({a_sig}), "
                    f".b({b_sig}), .sum({sum_sig}), .cout({carry1_sig}));\n"
                )
        f.write("\n")
        f.write("  // Final addition vectors\n")
        for col in range(C):
            f.write(f"  assign vec1[{col}] = {final_vec1[col]};\n")
            f.write(f"  assign vec2[{col}] = {final_vec2[col]};\n")
        carry_sigs = [
                         rec['output_carry1'] for rec in env.compression_history
                         if rec['output_carry1'] is not None and rec['column'] + 1 == C
                     ] + [
                         rec['output_carry2'] for rec in env.compression_history
                         if rec['output_carry2'] is not None and rec['column'] + 1 == C
                     ]
        carry_expr = " | ".join(carry_sigs) if carry_sigs else "1'b0"
        f.write(f"  assign vec1[{C}] = {carry_expr};\n")
        f.write(f"  assign vec2[{C}] = 1'b0;\n")
        f.write("\n")
        f.write("  assign product = vec1 + vec2;\n")
        f.write("endmodule\n")


def train():
    input_dim = 5  
    agent = DQNAgent(input_dim, NUM_COLUMNS)
    best_env = None
    best_cost = float('inf')
    cost_history = []
    ha_counts = []  
    fa_counts = []  
    comp53_counts = []  
    env_total_area = 0
    env_total_delay = 0
    t0 = time.time()

    cost_file = open("cost_history.txt", "w")
    cost_file.write("Episode,Cost\n")  

    for episode in range(1500):
        env = CompressionEnv()
        state = env.get_state()
        episode_transitions = []
        i = 2
        while True:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.act(state, valid)
            _, done = env.execute_action(action, i)
            next_state = env.get_state() if not done else None
            episode_transitions.append((state, action, 0, next_state, done, None))
            state = next_state
            i += 1
            if done:
                break

        print(f"Episode {episode} Over, generating Verilog and running synthesis...")
        generate_verilog(env, env.temp_verilog_file)
        total_area, total_delay = run_dc_synthesis(env.temp_verilog_file)

    
        ha_count = sum(1 for rec in env.compression_history if rec['action_type'] == 'ha')
        fa_count = sum(1 for rec in env.compression_history if rec['action_type'] == 'fa')
        comp53_count = sum(1 for rec in env.compression_history if rec['action_type'] == 'comp53')
        ha_counts.append(ha_count)
        fa_counts.append(fa_count)
        comp53_counts.append(comp53_count)

        prev_area = env_total_area
        prev_delay = env_total_delay
        env_total_area = total_area
        env_total_delay = total_delay

        prev_cost = prev_area * prev_delay if episode > 0 else float('inf')
        curr_cost = env_total_area * env_total_delay
        total_reward = prev_cost - curr_cost if episode > 0 else 0

        for transition in episode_transitions:
            state, action, _, next_state, done, _ = transition
            adjusted_reward = total_reward
            if action[1] == 'comp53':
                adjusted_reward *= 1
            if action[1] == 'fa':
                adjusted_reward *= 1
            agent.remember((state, action, adjusted_reward, next_state, done, None))

        agent.replay()
        cost_history.append(curr_cost)

        cost_file.write(f"{episode},{curr_cost:.1f}\n")
        cost_file.flush()  

        if len(cost_history) > 0:
            plt.clf()
            fig, ax1 = plt.subplots(figsize=(5, 5))
            color = 'black'
            ax1.set_xlabel('Episode', fontsize=12)
            ax1.set_ylabel('Compressor Count', color=color, fontsize=12)
            ax1.plot(ha_counts, label='Half Adder', color='#D14F44', linestyle='-', alpha=0.99)
            ax1.plot(fa_counts, label='Full Adder', color='#edb066', linestyle='-', alpha=0.99)
            ax1.plot(comp53_counts, label='5-3 Compressor', color='green', linestyle='-', alpha=0.99)
            ax1.tick_params(axis='y', labelcolor=color, size=10)

            
            fig.tight_layout()
            fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10)
            plt.savefig('training_curve_with_compressors.png')
            plt.close()  

        if curr_cost < best_cost:
            best_cost = curr_cost
            best_env = env

        print(
            f"Ep {episode}: current cost = {curr_cost:.1f}, HA Count = {ha_count}, FA Count = {fa_count}, 5-3 Count = {comp53_count}")

    if best_env is not None:
        t1 = time.time()
        generate_verilog(best_env, "multiplier_final.v")
        print(f"Final Verilog generated for best episode with cost {best_cost:.1f}")
        print(f"Total Time {t1 - t0}")

    
    cost_file.close()

if __name__ == "__main__":
    train()