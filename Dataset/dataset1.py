import pickle
import networkx as nx
from tqdm import tqdm
import pandas as pd
import functools
import pickle

def read_pkl(pkl_file):
    # Load data from a pkl file
    print(f'Reading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pkl(data, pkl_file):
    # Save data to a pkl file
    print(f'Saving data to {pkl_file}...')
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

def load_and_print_pkl(pkl_file):
    # Load a pkl file
    print(f'Loading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    
    # Print the first ten records of the data
    for i, transaction in enumerate(data):
        if i < 10:  # Only print the first ten records
            print(transaction)
        else:
            break

def extract_transactions(G):
    # Extract all transaction data from the graph
    transactions = []
    for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True), desc='accounts_data_generate'):
        amount = tnx_info['amount']
        block_timestamp = int(tnx_info['timestamp'])
        tag = G.nodes[from_address]['isp']
        transaction = {
            'tag': tag,
            'from_address': from_address,
            'to_address': to_address,
            'amount': amount,
            'timestamp': block_timestamp,
        }
        transactions.append(transaction)
    return transactions

def data_generate():
    graph_file = '/home/ducanhhh/Dynamic_Feature/Dataset/MulDiGraph.pkl'
    out_file = 'transactions1.pkl'
    
    # Read graph data
    graph = read_pkl(graph_file)
    # Extract transaction data
    transactions = extract_transactions(graph)
    # Save transaction data to a new file
    save_pkl(transactions, out_file)

if __name__ == '__main__':
    data_generate()
    pkl_file = 'transactions1.pkl'  # Make sure this file path is correct
    load_and_print_pkl(pkl_file)
