# BÁO CÁO BÀI TẬP: PHÂN CỤM VĂN BẢN TIẾNG VIỆT VNEXPRESS BẰNG K-MEANS

---

## 1. TẬP DỮ LIỆU (DATASET)

* **Tên dữ liệu**: Tập bài báo điện tử tiếng Việt thu thập từ VnExpress (`News_VNExpress_`).
* **Quy mô**: $N = 1,349$ bài báo.
* **Số lượng chủ đề thực tế (Ground Truth)**: Gồm 10 chuyên mục chính với phân bố không đều (từ 55 đến 263 bài/chủ đề):
  1. **Đời sống**: 121 bài
  2. **Du lịch**: 55 bài
  3. **Giải trí**: 202 bài
  4. **Giáo dục**: 106 bài
  5. **Khoa học**: 145 bài
  6. **Kinh doanh**: 263 bài
  7. **Pháp luật**: 60 bài
  8. **Sức khỏe**: 163 bài
  9. **Thể thao**: 174 bài
  10. **Thời sự**: 60 bài
* **Định dạng file dữ liệu**:
  * `clean_tfidf.npz`: Ma trận thưa TF-IDF ($1,349 \text{ bài báo} \times 24,355 \text{ từ vựng}$). Mật độ phần tử khác không: $0.5224\%$ (171,634 phần tử khác 0). Lưu trữ dạng `scipy.sparse.csr_matrix` chiếm ~2.06 MB (so với ~262.8 MB nếu lưu ma trận đặc).
  * `clean_labels.npy`: Mảng nhãn số nguyên $\{0, 1, \dots, 9\}$ tương ứng với 10 chủ đề thực tế của từng bài báo.
  * `clean_data.csv`: Dữ liệu văn bản sau bước tiền xử lý tách từ (`pyvi.ViTokenizer`) và lọc từ dừng (`vietnamese-stopwords.txt`).
* **Xử lý giảm chiều (LSA - Latent Semantic Analysis)**:
  * Do số chiều $24,355$ quá lớn gây ra hiện tượng bão hòa khoảng cách (Curse of Dimensionality), dữ liệu được giảm chiều bằng **TruncatedSVD** xuống **$50$ chiều ngữ nghĩa tiềm ẩn** (giải thích được $22.01\%$ tổng phương sai, gấp 107 lần so với kỳ vọng ngẫu nhiên).
  * Tiếp tục qua **Normalizer ($L_2$)** (`Normalizer(norm='l2', copy=False)`) để chuẩn hóa độ dài vector $\|x_i\|_2 = 1.0$, chiếu toàn bộ văn bản lên mặt cầu đơn vị $\mathbb{S}^{49}$.
  * **Định lý tương đương hình học**: Trên mặt cầu đơn vị $\mathbb{S}^{49}$, bình phương khoảng cách Euclidean tỷ lệ thuận nghiêm ngặt với Cosine Distance:
    $$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2\langle u, v \rangle = 1 + 1 - 2\cos(u, v) = 2 \cdot (1 - \cos(u, v)) = 2 \cdot d_{\cos}(u, v)$$
    Do đó, việc tối thiểu hóa tổng bình phương khoảng cách Euclidean trong K-Means hoàn toàn tương đương toán học với việc tối đa hóa Cosine Similarity (chuẩn mực cho ngữ nghĩa văn bản).

---

## 2. TẤT CẢ THÔNG SỐ ĐẦU VÀO CỦA K-MEANS (INPUTS & HYPERPARAMETERS)

| Tham số / Đầu vào | Giá trị thiết lập | Mục đích & Giải thích kỹ thuật |
| :--- | :--- | :--- |
| **Ma trận đầu vào (`X`)** | `X_lsa` ($1,349 \times 50$) | Ma trận đặc trưng 50 chiều thu được sau pipeline LSA (TruncatedSVD + L2 Normalizer). |
| **Số lượng cụm (`n_clusters` / `k`)** | $K = 10$ | Khớp đúng với 10 chủ đề báo chí thực tế của tập dữ liệu VnExpress. |
| **Hàm khoảng cách (Metric)** | Khoảng cách Euclidean trên vector chuẩn hóa $L_2$ | Khai triển nhị thức ma trận thuần NumPy (BLAS Level-3 GEMM): $\|x - c\|^2 = \|x\|^2 - 2XC^T + \|c\|^2$, không sử dụng thư viện ngoài `sklearn.metrics.pairwise`. |
| **Chiến lược khởi tạo (`init`)** | Khởi tạo ngẫu nhiên không hoàn lại (`replace=False`) | Chọn ngẫu nhiên $K=10$ mẫu bài báo trong $X$ làm tọa độ tâm ban đầu (`init='random'`, hỗ trợ cả `'k-means++'`). |
| **Số lần khởi tạo (`n_init`)** | $1$ | Thực hiện 1 lần khởi tạo độc lập (đảm bảo tính so sánh nhất quán với kết quả gốc của bài lab). |
| **Số vòng lặp tối đa (`max_iter`)** | $300$ | Giới hạn chặn trên số bước lặp luân phiên gán cụm - cập nhật tâm. |
| **Dung sai hội tụ (`tol`)** | $10^{-4}$ ($0.0001$) | Dừng thuật toán khi chuẩn Frobenius dịch chuyển của tâm cụm $\|\mu^{(t)} - \mu^{(t-1)}\|_F < tol$. |
| **Hạt giống ngẫu nhiên (`random_state`)** | $42$ | Đóng gói qua `np.random.RandomState(42)` độc lập, đảm bảo tái lập 100% mà không gây ô nhiễm seed toàn cục. |
| **Xử lý cụm rỗng (Empty Cluster Handling)** | Farthest-point heuristic | Khi cụm bị rỗng, gán tâm mới về mẫu có khoảng cách xa nhất so với tâm hiện tại ($\arg\max_i \min_k \|x_i - c_k\|^2$), giúp tối ưu hóa giảm hàm mục tiêu và loại bỏ cụm suy biến. |
| **Đồng bộ hóa nhãn (Label Sync)** | Post-convergence sync | Gán lại nhãn `self.labels_ = np.argmin(final_dist_sq, axis=1)` trên tâm cụm cuối cùng sau vòng lặp, đảm bảo tính nhất quán tuyệt đối giữa `predict(X)` và `labels_`. |

---

## 3. BẢNG KẾT QUẢ THU ĐƯỢC VÀ ĐỐI CHUẨN ĐỊNH LƯỢNG

### 3.1. Bảng so sánh định lượng chi tiết giữa MyKMeans (tự viết) và Sklearn KMeans

Dưới đây là bảng kết quả đối chuẩn khoa học được xuất trực tiếp từ mã nguồn thực thi của notebook:

| Chỉ số đánh giá | MyKMeans (Tự viết - Pure NumPy) | Sklearn KMeans (Thư viện chuẩn) | Nhận xét phân tích khoa học |
| :--- | :---: | :---: | :--- |
| **Thời gian thực thi (Runtime)** | **0.075s** | 0.533s | `MyKMeans` nhanh hơn gấp **7 lần** nhờ khai triển ma trận BLAS GEMM thuần túy và chạy `n_init=1`. |
| **Số vòng lặp hội tụ (`n_iter_`)** | **22 vòng** | 30 vòng | Hội tụ nhanh chóng với tiêu chí dung sai $tol = 10^{-4}$. |
| **Inertia (SSE - Tổng sai số bình phương)** | 902.07 | **879.03** | `Sklearn KMeans` đạt Inertia thấp hơn do sử dụng `k-means++` và chạy qua 10 lần khởi tạo độc lập (`n_init=10`). |
| **Độ tinh khiết phân cụm (Purity)** | **71.46%** | 63.97% | 71.46% số bài báo (964/1,349 bài) thuộc đúng chủ đề chiếm đa số tại cụm được gán. |
| **NMI (Normalized Mutual Information)** | **0.5858** | 0.5456 | Đo lường lượng thông tin chung giữa nhãn cụm và nhãn thực tế ($0 \to 1$). |
| **ARI (Adjusted Rand Index)** | **0.4767** | 0.4337 | Mức độ tương đồng cặp có hiệu chỉnh ngẫu nhiên (miền giá trị $[-1, 1]$, baseline ngẫu nhiên là 0). |
| **Silhouette Score** | 0.1181 | **0.1322** | Đo độ gắn kết nội cụm so với độ tách biệt láng giềng (miền giá trị $[-1, 1]$). |

---

### 3.2. Bảng đối chiếu các cấu hình thử nghiệm và kiểm chứng LSA

| Cấu hình thử nghiệm | Số chiều ($D$) | NMI (Thông tin tương hỗ) | ARI (Chỉ số Rand hiệu chỉnh) | Silhouette Score | Thời gian chạy | Đánh giá bản chất |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. TF-IDF gốc (Chưa giảm chiều)** | $24,355$ | $0.3512$ | $0.0821$ | $0.0064$ | $3.250\text{ s}$ | Bị lời nguyền số chiều, ma trận thưa 99.48%, khoảng cách bão hòa, cụm bị phân mảnh. |
| **2. SVD 50 chiều (Không chuẩn hóa $L_2$)** | $50$ | $0.0589$ | $0.0381$ | $0.0972$ | $0.185\text{ s}$ | Độ dài vector khác nhau làm sụp đổ cụm (ARI tụt dốc), chứng minh vai trò sống còn của L2 Normalizer. |
| **3. CẤU HÌNH TỐT NHẤT: LSA (SVD 50D + $L_2$ Normalizer) + MyKMeans** | **$50$** | **$\mathbf{0.5858}$** | **$\mathbf{0.4767}$** | **$\mathbf{0.1181}$** | **$\mathbf{0.075\text{ s}}$** | **Kết quả toàn diện nhất: NMI đạt gần 60%, ARI tăng gấp 6 lần so với TF-IDF gốc, Purity đạt 71.46%.** |
| **4. Đối chứng: LSA (SVD 50D + $L_2$) + Sklearn KMeans** | $50$ | $0.5456$ | $0.4337$ | $0.1322$ | $0.533\text{ s}$ | Đối chuẩn công nghiệp chuẩn mực từ Scikit-Learn với `k-means++` và `n_init=10`. |

---

### 3.3. Phân tích chi tiết độ tinh khiết (Purity) theo từng cụm của MyKMeans

Tổng độ tinh khiết toàn cục đạt **71.46%** (964/1,349 bài báo). Chi tiết phân bố từng cụm:
* **Cụm 0 (82 bài)**: **100.0%** bài thuộc chủ đề **Kinh doanh** (82 bài) — Độ tinh khiết tuyệt đối.
* **Cụm 2 (132 bài)**: **99.2%** bài thuộc chủ đề **Thể thao** (131/132 bài) — Gần như hoàn hảo.
* **Cụm 5 (164 bài)**: **87.2%** bài thuộc chủ đề **Kinh doanh** (143/164 bài).
* **Cụm 6 (140 bài)**: **82.9%** bài thuộc chủ đề **Khoa học** (116/140 bài).
* **Cụm 3 (134 bài)**: **81.3%** bài thuộc chủ đề **Sức khỏe** (109/134 bài).
* **Cụm 1 (76 bài)**: **80.3%** bài thuộc chủ đề **Giáo dục** (61/76 bài).
* **Cụm 8 (25 bài)**: **72.0%** bài thuộc chủ đề **Đời sống** (18/25 bài).
* **Cụm 4 (255 bài)**: **67.8%** bài thuộc chủ đề **Giải trí** (173/255 bài).
* **Cụm 7 (132 bài)**: **56.8%** bài thuộc chủ đề **Đời sống** (75/132 bài) xen kẽ cùng Pháp luật.
* **Cụm 9 (209 bài)**: **26.8%** bài thuộc chủ đề **Thời sự** (56 bài) — Cụm hỗn hợp tổng hợp các sự kiện tin tức xã hội.

---

### 3.4. Nhận xét phản biện khoa học khách quan

1. **Về việc MyKMeans đạt NMI và ARI cao hơn Sklearn KMeans trên seed 42**:
   - `MyKMeans` chỉ khởi tạo ngẫu nhiên 1 lần (`n_init=1`), trong khi `Sklearn KMeans` chạy qua 10 lần khởi tạo độc lập (`n_init=10`) với chiến lược chọn tâm `k-means++`.
   - Về mặt toán học tối ưu hàm mục tiêu Inertia (SSE), Sklearn đạt $879.03$ (thấp hơn, tức là tối ưu tốt hơn nghiệm của MyKMeans đạt $902.07$).
   - Việc `MyKMeans` đạt NMI ($0.5858$) và ARI ($0.4767$) cao hơn Sklearn trên `seed=42` là một **hiện tượng ngẫu nhiên thuận lợi tại cực tiểu địa phương cụ thể**, khi ranh giới phân hoạch vô tình trùng khớp chặt chẽ hơn với nhãn nhân tạo của VnExpress.
2. **Về đồ thị phân bố t-SNE 2D**:
   - Chiếu không gian 50 chiều về mặt phẳng 2D bằng t-SNE (`perplexity=30`) bảo toàn trực quan cấu trúc hình học: các chủ đề có từ vựng đặc thù (*Thể thao*, *Kinh doanh*, *Khoa học*, *Sức khỏe*) hình thành các cụm đảo cô lập, tách biệt rõ ràng.
   - Các chủ đề tổng hợp xã hội (*Thời sự*, *Đời sống*, *Pháp luật*) nằm xen kẽ ở vùng trung gian giữa các cụm, phản ánh đúng bản chất giao thoa ngữ nghĩa của báo chí thực tế.
