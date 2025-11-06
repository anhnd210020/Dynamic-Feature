import pickle
import random
import numpy as np

# Load data from file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Classify normal and abnormal accounts and load transaction data
data_filename = 'transactions4.pkl'
accounts_data = load_data(data_filename)

normal_accounts = {}
abnormal_accounts = {}

for address, transactions in accounts_data.items():
    if transactions[0]['tag'] == 0:
        normal_accounts[address] = transactions
    elif transactions[0]['tag'] == 1:
        abnormal_accounts[address] = transactions

# Get number of abnormal accounts
num_abnormal = len(abnormal_accounts)

# Randomly select twice the number of abnormal accounts from normal accounts
selected_normal_accounts = random.sample(list(normal_accounts.keys()), 2 * num_abnormal)
adjusted_normal_accounts = {addr: normal_accounts[addr] for addr in selected_normal_accounts}

# Merge adjusted normal accounts with all abnormal accounts
adjusted_accounts_data = {**adjusted_normal_accounts, **abnormal_accounts}

# Save adjusted data
save_data_filename = 'adjusted_transactions4.pkl'
save_data(adjusted_accounts_data, save_data_filename)

print(f"Data has been adjusted and saved to {save_data_filename}")
print(f"Number of abnormal accounts: {len(abnormal_accounts)}")
print(f"Number of selected normal accounts: {len(adjusted_normal_accounts)}")

# Print the first ten transactions of the first ten accounts
print("\nFirst ten accounts with their first ten transaction records:")
for address in list(adjusted_accounts_data.keys())[:10]:  # Only display data for the first ten accounts
    print(f"\nAccount {address} - First ten transactions:")
    for transaction in adjusted_accounts_data[address][:10]:  # Show the first ten records for each account
        print(transaction)

# Define weight calculation function
def calculate_weight(transaction):
    weights = []
    if '2-gram' in transaction:
        weights.append(transaction['2-gram'] * 0.1)
    if '3-gram' in transaction:
        weights.append(transaction['3-gram'] * 0.2)
    if '4-gram' in transaction:
        weights.append(transaction['4-gram'] * 0.3)
    if '5-gram' in transaction:
        weights.append(transaction['5-gram'] * 0.4)
    return np.sum(weights) if weights else 0  # Compute average; if the list is empty, return 0

# Extract all unique account addresses, only include currently remaining accounts
addresses = set(adjusted_accounts_data.keys())

# Mapping from address to index
address_to_index = {addr: idx for idx, addr in enumerate(addresses)}

# Create adjacency matrix
n = len(addresses)
adj_matrix = np.zeros((n, n), dtype=float)  # Use float type to store weights
# Save mapping from address to index
save_data(address_to_index, 'data_Dataset.address_to_index')
# Fill adjacency matrix
for account, transactions in adjusted_accounts_data.items():
    for transaction in transactions:
        from_addr = transaction['from_address']
        to_addr = transaction['to_address']
        if from_addr in addresses and to_addr in addresses:
            from_idx = address_to_index[from_addr]
            to_idx = address_to_index[to_addr]
            weight = calculate_weight(transaction)  # Compute weight
            adj_matrix[from_idx, to_idx] += weight  # Accumulate weight

# Save adjacency matrix
save_data(adj_matrix, 'weighted_adjacency_matrix.pkl')
