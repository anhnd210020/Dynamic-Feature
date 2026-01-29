# #ETH-GBert
# import subprocess
# import os
# import shutil

# def run_script(script_name):
#     try:
#         print(f"Running {script_name}...")
#         subprocess.run(['python', script_name], check=True)
#         print(f"{script_name} completed successfully.\n")
#     except subprocess.CalledProcessError as e:
#         print(f"Error occurred while running {script_name}: {e}")


# import os
# import shutil

# def move_files_to_preprocessed_folder():
#     # Correct the folder name if it's a typo (change 'preprocesse' to 'preprocessed' if needed)
#     destination_folder = '../data/preprocessed/Dataset'  # Fix typo here if it's 'preprocesse' in your code
    
#     # Create destination folder if it doesn't exist
#     os.makedirs(destination_folder, exist_ok=True)
    
#     files_to_move = [
#         'data_Dataset.address_to_index',
#         'data_Dataset.labels',
#         'data_Dataset.shuffled_clean_docs',
#         'data_Dataset.test_y',
#         'data_Dataset.test_y_prob',
#         'data_Dataset.tfidf_list',
#         'data_Dataset.train_y',
#         'data_Dataset.train_y_prob',
#         'data_Dataset.valid_y',
#         'data_Dataset.valid_y_prob',
#         'data_Dataset.y',
#         'data_Dataset.y_prob',
#         'dev.tsv',   # This is the one causing the error
#         'test.tsv',
#         'train.tsv'  # Add if needed, based on your script
#     ]
    
#     for file_name in files_to_move:
#         if os.path.exists(file_name):
#             dest_path = os.path.join(destination_folder, os.path.basename(file_name))
#             if os.path.exists(dest_path):
#                 print(f"Overwriting existing file: {dest_path}")
#                 os.remove(dest_path)  # Remove existing to allow overwrite
#             shutil.move(file_name, destination_folder)
#             print(f"Moved {file_name} to {destination_folder}")
#         else:
#             print(f"{file_name} does not exist and will not be moved.")
            
# if __name__ == '__main__':
#     for i in range(1, 12):
#         script_name = f"dataset{i}.py"
#         run_script(script_name)
#     run_script("adjust_matrix.py")
#     run_script("BERT_text_data.py")
#     move_files_to_preprocessed_folder()
    
    
#ETH-GSetBert
import subprocess
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

def move_files_to_preprocessed_folder():
    # Correct the folder name if it's a typo (change 'preprocesse' to 'preprocessed' if needed)
    destination_folder = '../data/preprocessed/Dataset'  # Fix typo here if it's 'preprocesse' in your code
    
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    files_to_move = [
        'data_Dataset.address_to_index',
        'data_Dataset.labels',
        'data_Dataset.shuffled_clean_docs',
        'data_Dataset.test_y',
        'data_Dataset.test_y_prob',
        'data_Dataset.tfidf_list',
        'data_Dataset.train_y',
        'data_Dataset.train_y_prob',
        'data_Dataset.valid_y',
        'data_Dataset.valid_y_prob',
        'data_Dataset.y',
        'data_Dataset.y_prob',
        'dev.tsv',   # This is the one causing the error
        'test.tsv',
        'train.tsv'  # Add if needed, based on your script
    ]
    
    for file_name in files_to_move:
        if os.path.exists(file_name):
            dest_path = os.path.join(destination_folder, os.path.basename(file_name))
            if os.path.exists(dest_path):
                print(f"Overwriting existing file: {dest_path}")
                os.remove(dest_path)  # Remove existing to allow overwrite
            shutil.move(file_name, destination_folder)
            print(f"Moved {file_name} to {destination_folder}")
        else:
            print(f"{file_name} does not exist and will not be moved.")

import numpy as np
import pickle
from env_config import env_config

PREP_DIR = '../data/preprocessed/Dataset'

def load_address_to_index(path=os.path.join(PREP_DIR, 'data_Dataset.address_to_index')):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def try_load_transactions():
    """
    Trả về dict: addr(str) -> list of tx dicts, mỗi tx:
      {
        "amount": float,
        "timestamp": int|float (seconds),
        "in_out": int (1=in, 0=out),
        "token_id": int (tuỳ bạn mapping)
      }
    Nếu KHÔNG có dữ liệu giao dịch, trả về {} để fallback sang zeros.
    Bạn có thể thay đường dẫn/định dạng tại đây (pkl/csv/parquet tuỳ bạn).
    """
    candidates = [
        './transactions.pkl',
        '../data/raw/transactions.pkl',
        os.path.join(PREP_DIR, 'transactions.pkl'),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    print(f'Loaded transactions from {p}')
                    return pickle.load(f)
            except Exception as e:
                print(f'Failed to load {p}: {e}')
    print('No transaction file found. Will create zero set_feats/set_mask as placeholder.')
    return {}

def build_tx_features(txs, d_in):
    """
    txs: list đã sort theo thời gian ASC (nếu chưa, ta sort ở hàm caller)
    Trả về np.ndarray [len(txs), d_in].
    Định nghĩa cột (gợi ý tối thiểu):
      0: amount_log = log1p(amount)
      1..4: Δt2..Δt5 (giây) (0 nếu không đủ lịch sử)
      5: in_out (0/1)
      6: token_id (int hoặc id đã chuẩn hoá)
    Các cột còn lại (nếu d_in>7) sẽ zero-pad.
    Bạn có thể sửa/ mở rộng tuỳ theo schema thật.
    """
    if len(txs) == 0:
        return np.zeros((0, d_in), dtype=np.float32)

    # sort theo timestamp
    txs = sorted(txs, key=lambda x: x.get("timestamp", 0))

    # vector hoá
    ts = np.array([t.get("timestamp", 0) for t in txs], dtype=np.float64)
    amt = np.array([t.get("amount", 0.0) for t in txs], dtype=np.float64)
    inout = np.array([t.get("in_out", 0) for t in txs], dtype=np.float32)
    token_id = np.array([t.get("token_id", 0) for t in txs], dtype=np.float32)

    # features cơ bản
    amount_log = np.log1p(np.maximum(amt, 0.0)).astype(np.float32)

    # Δt n-gram
    def delta_n(ts, n):
        out = np.zeros_like(ts, dtype=np.float32)
        out[n-1:] = (ts[n-1:] - ts[:-n+1])  # seconds
        return out
    dt2 = delta_n(ts, 2)
    dt3 = delta_n(ts, 3)
    dt4 = delta_n(ts, 4)
    dt5 = delta_n(ts, 5)

    # ghép thành [len, 7] rồi pad nếu d_in>7
    base = np.stack([amount_log, dt2, dt3, dt4, dt5, inout, token_id], axis=1).astype(np.float32)
    if d_in > base.shape[1]:
        pad = np.zeros((base.shape[0], d_in - base.shape[1]), dtype=np.float32)
        base = np.concatenate([base, pad], axis=1)
    elif d_in < base.shape[1]:
        base = base[:, :d_in]
    return base.astype(np.float32, copy=False)

def build_set_banks(address_to_index, tx_index, Lmax, d_in, save_dir=PREP_DIR):
    """
    address_to_index: dict addr->idx
    tx_index: dict addr->list[tx]
    Lmax, d_in: theo env_config
    Lưu: set_feats.npy [N,Lmax,d_in], set_mask.npy [N,Lmax]
    """
    N = len(address_to_index)
    set_feats = np.zeros((N, Lmax, d_in), dtype=np.float32)
    set_mask  = np.zeros((N, Lmax), dtype=bool)

    # duyệt theo index để khớp thứ tự tuyệt đối
    for addr, idx in address_to_index.items():
        txs = tx_index.get(addr, [])
        feats = build_tx_features(txs, d_in)  # [len(txs), d_in]

        if feats.shape[0] >= Lmax:
            # lấy gần nhất Lmax giao dịch
            feats_cut = feats[-Lmax:, :]
            mask_cut = np.ones((Lmax,), dtype=bool)
        else:
            pad_len = Lmax - feats.shape[0]
            feats_cut = np.vstack([feats, np.zeros((pad_len, d_in), dtype=np.float32)])
            mask_cut = np.concatenate([np.ones((feats.shape[0],), dtype=bool),
                                       np.zeros((pad_len,), dtype=bool)])
        set_feats[idx] = feats_cut
        set_mask[idx] = mask_cut

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'set_feats.npy'), set_feats)
    np.save(os.path.join(save_dir, 'set_mask.npy'), set_mask)
    print(f'Saved set_feats.npy {set_feats.shape} and set_mask.npy {set_mask.shape} to {save_dir}')

def generate_set_features():
    # load mapping
    address_to_index = load_address_to_index()
    Lmax = env_config.SET_LMAX
    d_in = env_config.SET_D_IN

    # cố gắng nạp dữ liệu giao dịch; nếu không có -> rỗng (fallback zeros)
    tx_index = try_load_transactions()
    build_set_banks(address_to_index, tx_index, Lmax, d_in, save_dir=PREP_DIR)
                
if __name__ == '__main__':
    # chạy các bước cũ
    for i in range(1, 12):
        script_name = f"dataset{i}.py"
        run_script(script_name)
    run_script("adjust_matrix.py")
    run_script("BERT_text_data.py")
    move_files_to_preprocessed_folder()

    # NEW: tạo set_feats / set_mask (an toàn cả khi chưa có dữ liệu giao dịch)
    generate_set_features()





