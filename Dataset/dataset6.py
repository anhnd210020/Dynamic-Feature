import pickle
import random
import tqdm

# Load data from file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Shuffle the order of transaction data within each account
def shuffle_transactions(accounts):
    for address in tqdm.tqdm(accounts.keys(), desc="Shuffling transactions"):
        random.shuffle(accounts[address])

# Load data
accounts_data = load_data('transactions5.pkl')

# Shuffle transaction data
shuffle_transactions(accounts_data)

# Save data
save_data(accounts_data, 'transactions6.pkl')

# Print the first five processed transactions for each account
print("Print the first five processed transactions for each account:")
for address in list(accounts_data.keys())[:5]:  # Only display data for the first five accounts
    print(f"Account {address} - First five transactions:")
    for transaction in accounts_data[address][:5]:  # Show the first five records for each account
        print(transaction)
    print("\n")

print("Transaction data has been shuffled and saved to transactions6.pkl.")