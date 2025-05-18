# Comment Sentiment Analysis

Ứng dụng phân tích cảm xúc comment sử dụng nhiều mô hình Machine Learning và Deep Learning, giao diện Streamlit hiện đại, sáng tạo.

## 🚀 Hướng dẫn cài đặt & deploy trên Streamlit Community Cloud

### 1. Chuẩn bị
- Đảm bảo các file sau có trong repository:
  - `app.py` (file chính chạy Streamlit)
  - `requirements.txt` (các thư viện cần thiết)
  - Thư mục `weights/` chứa các file model: `k_means.pkl`, `random_forest.pkl`, `logistic_regression.pkl`, `knn.pkl.gz`, `svm.pkl.gz`, `lstm.h5`
  - Thư mục `data/` chứa dữ liệu: `processed_train.csv`, `processed_test.csv`

### 2. Deploy lên Streamlit Community Cloud
1. Đẩy toàn bộ mã nguồn lên GitHub.
2. Truy cập https://share.streamlit.io/ và đăng nhập bằng GitHub.
3. Chọn repository chứa dự án, chọn file `app.py`.
4. Nhấn "Deploy".
5. Chờ vài phút, ứng dụng sẽ có link public.

### 3. Chạy thử local (tùy chọn)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 4. Lưu ý
- Các file model và dữ liệu phải có trong repo hoặc upload thủ công lên Streamlit Cloud.
- Nếu app báo thiếu file, kiểm tra lại đường dẫn và tên file.

---

**Chúc bạn deploy thành công!**
