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

# Select and shuffle accounts
def select_and_shuffle_accounts(accounts):
    tag1_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 1]
    tag0_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 0]
    
    # Randomly select accounts with tag = 0, the number is twice the number of tag = 1 accounts
    double_tag1_count = random.sample(tag0_accounts, 2 * len(tag1_accounts))
    
    # Combine and shuffle the order
    selected_accounts = tag1_accounts + double_tag1_count
    random.shuffle(selected_accounts)
    
    # Return shuffled dictionary
    return dict(selected_accounts)

# Load data
accounts_data = load_data('transactions7.pkl')

# Select and shuffle accounts
shuffled_accounts_data = select_and_shuffle_accounts(accounts_data)

# Save data
save_data(shuffled_accounts_data, 'transactions8.pkl')

# Print the first ten processed accounts
print("Print the first ten accounts:")
for address, transactions in list(shuffled_accounts_data.items())[:10]:  # Only display data for the first ten accounts
    print(f"Account {address}:")
    print(transactions)
    print("\n")

print("Data has been processed and saved to transactions8.pkl.")






