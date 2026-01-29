# # ETH-GBert.py
# import os
# from pathlib import Path

# from dotenv import load_dotenv


# class EnvConfig:
#     env_path = Path(__file__).parent.parent.parent / ".env"
#     if env_path.exists():
#         load_dotenv(dotenv_path=env_path)

#     GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", 44))
#     TRANSFORMERS_OFFLINE = int(os.environ.get("TRANSFORMERS_OFFLINE", 0))
#     HUGGING_LOCAL_MODEL_FILES_PATH = os.environ.get(
#         "HUGGING_LOCAL_MODEL_FILES_PATH", ""
#     )


# env_config = EnvConfig()



#ETH-GSetBert
import os
from pathlib import Path

from dotenv import load_dotenv

class EnvConfig:
    # giữ nguyên cách tìm .env như hiện tại
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    # ====== hiện có ======
    GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", 44))
    TRANSFORMERS_OFFLINE = int(os.environ.get("TRANSFORMERS_OFFLINE", 0))
    HUGGING_LOCAL_MODEL_FILES_PATH = os.environ.get("HUGGING_LOCAL_MODEL_FILES_PATH", "")

    # ====== BỔ SUNG: cấu hình SetTransformer (phần mở rộng) ======
    # Kích thước đặc trưng 1 giao dịch (amount, Δt n-gram, in/out, token type, ...)
    SET_D_IN = int(os.environ.get("SET_D_IN", 16))
    # Chiều mô hình bên trong SAB/PMA
    SET_D_MODEL = int(os.environ.get("SET_D_MODEL", 128))
    SET_HEADS = int(os.environ.get("SET_HEADS", 4))
    SET_LAYERS = int(os.environ.get("SET_LAYERS", 2))
    SET_DROPOUT = float(os.environ.get("SET_DROPOUT", 0.1))
    # Số giao dịch tối đa mỗi account (pad/truncation)
    SET_LMAX = int(os.environ.get("SET_LMAX", 128))

    # ====== BỔ SUNG: cấu hình Fusion sau pooler (3 nhánh) ======
    FUSION_HIDDEN = int(os.environ.get("FUSION_HIDDEN", 256))
    GUMBEL_TAU = float(os.environ.get("GUMBEL_TAU", 1.0))
    # 0/1 -> bool
    GUMBEL_HARD = bool(int(os.environ.get("GUMBEL_HARD", 0)))
    # Khởi tạo α cho O3: w = sigmoid(α)
    FUSION_ALPHA_INIT = float(os.environ.get("FUSION_ALPHA_INIT", 0.5))

    # ====== Tùy chọn tiện dụng ======
    # Cho phép đổi nhanh tên BERT qua env (nếu bạn cần)
    BERT_NAME = os.environ.get("BERT_NAME", "")
    # Nếu đặt 1: tắt nhánh GCN-token trong embeddings (dùng cho ablation)
    GCN_DISABLE_IN_EMB = bool(int(os.environ.get("GCN_DISABLE_IN_EMB", 0)))


env_config = EnvConfig()