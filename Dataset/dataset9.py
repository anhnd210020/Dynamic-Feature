import pickle
import tqdm

# Load data from file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Remove 'tag' field from transactions
def remove_tag_from_transactions(accounts):
    for address, transactions in accounts.items():
        for transaction in transactions:
            for sub_transaction in transaction['transactions']:
                if 'tag' in sub_transaction:
                    del sub_transaction['tag']

# Load data
accounts_data = load_data('transactions8.pkl')

# Remove 'tag' field
remove_tag_from_transactions(accounts_data)

# Save data
save_data(accounts_data, 'transactions9.pkl')

# Print the first ten accounts' data
print("Print the first ten accounts:")
for address, transactions in list(accounts_data.items())[:10]:  # Only display data for the first ten accounts
    print(f"Account {address}:")
    for transaction in transactions:
        print(transaction)
    print("\n")

print("The 'tag' field has been removed and the data has been saved to transactions9.pkl.")






