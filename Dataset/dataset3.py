import pickle

# Load data from file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Sort each account's transaction data by timestamp
def sort_transactions_by_timestamp(accounts):
    sorted_accounts = {}
    for address, transactions in accounts.items():
        sorted_accounts[address] = sorted(transactions, key=lambda x: x['timestamp'])
    return sorted_accounts

# Load data
accounts_data = load_data('transactions2.pkl')

# Sort data
sorted_accounts_data = sort_transactions_by_timestamp(accounts_data)

# Print the first ten sorted transactions for each account
print("Print the first ten sorted transactions for each account:")
for address in list(sorted_accounts_data.keys())[:10]:  # Only display data for the first ten accounts
    print(f"Account {address} - First ten transactions:")
    for transaction in sorted_accounts_data[address][:10]:  # Show the first ten records for each account
        print(transaction)
    print("\n")

# Save data
save_data(sorted_accounts_data, 'transactions3.pkl')

print("Data has been sorted by timestamp for each account and saved to transactions3.pkl.")






