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

# Remove specific fields
def remove_fields(accounts, fields):
    for address in tqdm.tqdm(accounts.keys(), desc="Removing fields"):
        for transaction in accounts[address]:
            for field in fields:
                if field in transaction:
                    del transaction[field]

# Load data
accounts_data = load_data('transactions4.pkl')

# Fields to remove
fields_to_remove = ['from_address', 'to_address', 'timestamp']

# Remove fields
remove_fields(accounts_data, fields_to_remove)

# Save data
save_data(accounts_data, 'transactions5.pkl')

# Print the first ten processed transactions for each account
print("Print the first ten processed transactions for each account:")
for address in list(accounts_data.keys())[:10]:  # Only display data for the first ten accounts
    print(f"Account {address} - First ten transactions:")
    for transaction in accounts_data[address][:10]:  # Show the first ten records for each account
        print(transaction)
    print("\n")

print("Fields have been removed and data saved to transactions5.pkl.")






