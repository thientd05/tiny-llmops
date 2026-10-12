# Quy tắc viết báo cáo + khung báo cáo

## Quy tắc viết

1. **Viết cho người hiểu kỹ thuật cơ bản nhưng chưa đọc code.** Giải thích cơ chế thật (cái gì chạy, tốn bộ nhớ/thời gian ở đâu),
   không rút gọn thành "vì nó tối ưu hơn".
2. **Không dùng ví von, hình ảnh so sánh** (không "như đường cao tốc", "như nút cổ chai"...). Nói thẳng con số và cơ chế.
3. **Không dùng từ mới lạ khi có từ thường.** Thuật ngữ bắt buộc phải dùng → giải thích ngay lần đầu xuất hiện trong báo cáo
   (một câu, trong ngoặc hoặc ngay sau), và thêm vào [../thuat-ngu.md](../thuat-ngu.md) nếu chưa có.
4. **Mỗi kết luận đi kèm số liệu và nguồn** (file kết quả hoặc mục trong measurements.md). Không có số → ghi rõ là phỏng đoán.
5. **Tách rõ "đo được" và "suy ra".** Ví dụ: "đo được KV cache chứa 36k token" khác "suy ra tối đa ~10 user RAG cùng lúc".
6. **Câu ngắn, mỗi đoạn một ý.** Bảng cho số liệu, chữ cho giải thích.
7. Chuyện không như dự kiến (lỗi, phải đổi cách làm) ghi thật, kể cả khi là do mình làm sai.

## Khung báo cáo (`bao-cao.md`)

```markdown
# Báo cáo P0X — <tên plan>

> Ngày làm: ... · Máy: host ... / machine ... · vLLM ... · Giờ GPU: ... (≈ ... VND)

## 1. Tóm tắt (3–5 dòng)
Làm gì, kết quả chính, có đạt tiêu chí xong của plan không.

## 2. Đã làm gì và vì sao
Các bước thực tế (khác plan ở đâu thì nói rõ và lý do).

## 3. Kết quả
Bảng số liệu chính. Link tới file kết quả thô và mục measurements.md.

## 4. Giải thích kết quả
Vì sao ra số này: phần nào tốn thời gian/bộ nhớ, điều gì giới hạn.
Phân biệt rõ: đo được / suy ra / phỏng đoán.

## 5. Vấn đề gặp phải
Triệu chứng → nguyên nhân → cách xử lý. Vấn đề lớn → link challenges.md.

## 6. Ảnh hưởng tới bước sau
Điều gì cần đổi trong các plan tiếp theo; ý tưởng tune / nghi vấn điểm nghẽn rút ra được.

## 7. Thuật ngữ dùng trong báo cáo này
Danh sách thuật ngữ + giải thích một dòng (chỉ những từ đã dùng ở trên).
```
