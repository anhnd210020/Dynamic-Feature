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

# Remove the 'tag' field from all transactions except the first one
def remove_tag_except_first(accounts):
    for address, transactions in accounts.items():
        for i in range(1, len(transactions)):
            if 'tag' in transactions[i]:
                del transactions[i]['tag']

# Merge all transactions of each account into a single entry
def merge_transactions(accounts):
    for address in accounts.keys():
        if accounts[address]:
            first_tag = accounts[address][0]['tag']  # Keep the tag of the first transaction
            merged_data = {'tag': first_tag, 'transactions': accounts[address]}
            accounts[address] = [merged_data]

# Load data
accounts_data = load_data('transactions6.pkl')

# Remove tag field
remove_tag_except_first(accounts_data)

# Merge transaction data
merge_transactions(accounts_data)

# Save data
save_data(accounts_data, 'transactions7.pkl')

# Print the first ten processed transactions for each account
print("Print the first ten processed transactions for each account:")
for address in list(accounts_data.keys())[:10]:  # Only display data for the first ten accounts
    print(f"Account {address} - First ten transactions:")
    for transaction in accounts_data[address][:10]:  # Show the first ten records for each account
        print(transaction)
    print("\n")

print("Transaction data has been processed and saved to transactions7.pkl.")
