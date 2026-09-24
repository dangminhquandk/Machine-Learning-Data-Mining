# Machine Learning & Data Mining (Học máy & Khai phá dữ liệu)

Kho lưu trữ tài liệu bài giảng lý thuyết và toàn bộ mã nguồn bài thực hành môn **Học máy & Khai phá dữ liệu**.

---

## 📂 Cấu trúc Repository

```text
├── Slide/                                  # Trọn bộ slide bài giảng lý thuyết
└── Buoi thuc hanh so 1/                    # Buổi thực hành số 1
    ├── Practice/                           # Bài thực hành trên lớp (Giá nhà Bangalore)
    │   ├── Bangalore_House_Price_data/     # Dữ liệu gốc & dữ liệu đã làm sạch
    │   ├── Data_Preprocessing.ipynb        # Tiền xử lý, lọc 3 tầng ngoại lai
    │   └── Model_Train_Test.ipynb          # Huấn luyện mô hình hồi quy (Linear, Lasso, Ridge, SVR, Random Forest)
    └── Assignment/                         # Bài tập về nhà
        └── News_VNExpress_/
            ├── Preprocessing_News.ipynb    # Tiền xử lý văn bản tiếng Việt & biểu diễn vector TF-IDF
            ├── vietnamese-stopwords.txt    # Danh từ dừng tiếng Việt
            └── news_vnexpress/             # 1.349 bài báo thuộc 10 chủ đề
```

---

## 🛠️ Hướng dẫn cài đặt môi trường (Conda)

```zsh
# 1. Tạo môi trường ảo
conda create -n course python=3.10 -y
conda activate course

# 2. Cài đặt các thư viện cần thiết
pip install -r "Buoi thuc hanh so 1/requirements.txt" ipykernel
```
