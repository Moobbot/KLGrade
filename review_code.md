# 1. Tổng quan các file chính

| Tên file                  | Chức năng dự đoán |
|---------------------------|-------------------|
| main.py                   | Điểm khởi đầu, chạy pipeline chính của dự án |
| model.py                  | Định nghĩa các mô hình học máy/deep learning |
| dataset.py                | Xử lý, load, chuẩn bị dữ liệu đầu vào |
| my_custom_loss.py         | Hàm loss tự định nghĩa cho mô hình |
| early_stop.py             | Cơ chế dừng sớm khi huấn luyện |
| ensemble_combinations.py  | Kết hợp nhiều mô hình (ensemble) |
| cam.py, cam_ensemble.py   | Class Activation Mapping (CAM), có thể dùng cho interpretability |
| cam_pyplot.py             | Vẽ biểu đồ liên quan đến CAM |
| box_plot.py               | Vẽ boxplot cho kết quả hoặc dữ liệu |
| make_csv.py               | Tạo file CSV từ dữ liệu/kết quả |
| score.py, score_auto.py, score_auto_all.py | Đánh giá mô hình, tự động hóa quá trình scoring |
| test_auto.py, test_ensemble.py | Tự động test, test cho mô hình ensemble |
| write_performance.py      | Ghi lại hiệu năng mô hình |

# 2. Luồng hoạt động dự đoán

1. **main.py**: Có thể là file chính để chạy huấn luyện, test, hoặc pipeline tổng thể.
2. **dataset.py**: Đọc và chuẩn bị dữ liệu, có thể trả về DataLoader hoặc numpy array.
3. **model.py**: Định nghĩa kiến trúc mô hình, có thể là CNN, RNN hoặc các mô hình deep learning khác.
4. **my_custom_loss.py**: Định nghĩa hàm loss đặc biệt, có thể dùng cho bài toán đặc thù.
5. **early_stop.py**: Theo dõi quá trình huấn luyện, dừng lại khi không còn cải thiện.
6. **ensemble_combinations.py**: Kết hợp nhiều mô hình để tăng hiệu quả dự đoán.
7. **score.py, score_auto.py, score_auto_all.py**: Đánh giá mô hình, có thể tính các chỉ số như accuracy, F1, AUC,...
8. **test_auto.py, test_ensemble.py**: Tự động hóa quá trình kiểm thử, đặc biệt cho các mô hình ensemble.
9. **cam.py, cam_ensemble.py, cam_pyplot.py**: Sinh và trực quan hóa bản đồ kích hoạt (CAM) để giải thích mô hình.
10. **box_plot.py, write_performance.py, make_csv.py**: Hỗ trợ trực quan hóa, ghi lại kết quả, xuất dữ liệu.
