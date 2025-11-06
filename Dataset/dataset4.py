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

# Add n-gram data to each transaction
def add_n_grams(accounts):
    for address, transactions in tqdm.tqdm(accounts.items(), desc="Processing n-gram data"):
        for n in range(2, 6):  # Process 2-gram to 5-gram
            gram_key = f"{n}-gram"
            for i in range(len(transactions)):
                if i < n-1:
                    transactions[i][gram_key] = 0  # Set initial n-1 values to 0
                else:
                    transactions[i][gram_key] = transactions[i]['timestamp'] - transactions[i-n+1]['timestamp']

# Load data
accounts_data = load_data('transactions3.pkl')

# Add n-gram data
add_n_grams(accounts_data)

# Save data
save_data(accounts_data, 'transactions4.pkl')

# Print the first ten processed transactions for each account
print("Print the first ten processed transactions for each account:")
for address in list(accounts_data.keys())[:10]:  # Only display data for the first ten accounts
    print(f"Account {address} First ten transactions:")
    for transaction in accounts_data[address][:10]:  # Show the first ten records for each account
        print(transaction)
    print("\n")

print("n-gram calculations have been completed and saved to transactions4.pkl.")






