# Quy tắc viết báo cáo + khung báo cáo

Mỗi plan xong có **hai** báo cáo, đặt cạnh `plan.md`:

| File | Cho ai, khi nào | Kiểu viết |
|---|---|---|
| `doc-ngay.md` | User **đọc ngay** sau khi plan xong | Kể lại từng bước theo thời gian, kèm lệnh thật và giải thích từng phần. Đọc đến đâu hiểu đến đó, không phải tra từ |
| `bao-cao.md` | User đọc sau, khi cần đi sâu; tư liệu bảo vệ | Đầy đủ số liệu, phân tích, đo được / suy ra / phỏng đoán |

Claude không cần đọc lại `doc-ngay.md` khi làm việc: mọi số liệu đã có trong `bao-cao.md` và measurements.md.

## Quy tắc viết `doc-ngay.md`

1. **Mở đầu bằng "Một câu"**: làm gì, tốn bao nhiêu, kết quả chính.
2. **Kể theo đúng trình tự đã làm** (bước 0, 1, 2...). Mỗi bước: làm gì → lệnh thật → giải thích → kết quả thật.
3. **Lệnh nào cũng giải thích từng phần**: bảng "phần → nghĩa", hoặc chú thích `←` ngay cạnh dòng lệnh.
4. **Từ chuyên môn giải thích ngay tại chỗ xuất hiện đầu tiên**, trong câu, không bắt người đọc mở bảng thuật ngữ.
   Từ nào có từ thường thay được thì dùng từ thường ("câu hỏi" thay "request", "bộ nhớ GPU" thay "VRAM").
5. **Không ví von.** Giải thích bằng cơ chế thật và con số thật.
6. **Lỗi và chỗ phải làm lại cũng kể**: thông báo lỗi thật → nghĩa là gì → sửa thế nào.
7. **Cuối bài có**: dòng thời gian của phiên (nếu thuê máy), chi phí theo hoá đơn, "Rút ra điều gì", "Bước tiếp theo".
8. Đọc riêng file này là hiểu được, không cần mở file khác.

## Quy tắc viết `bao-cao.md`

1. **Viết cho người hiểu kỹ thuật cơ bản nhưng chưa đọc code.** Giải thích cơ chế thật (cái gì chạy, tốn bộ nhớ/thời gian ở đâu),
   không rút gọn thành "vì nó tối ưu hơn".
2. **Không dùng ví von, hình ảnh so sánh** (không "như đường cao tốc", "như nút cổ chai"...). Nói thẳng con số và cơ chế.
3. **Không dùng từ mới lạ khi có từ thường.** Thuật ngữ bắt buộc phải dùng → giải thích ngay lần đầu xuất hiện trong báo cáo
   (một câu, trong ngoặc hoặc ngay sau), và thêm vào [../thuat-ngu.md](../thuat-ngu.md) nếu chưa có.
4. **Mỗi kết luận đi kèm số liệu và nguồn** (file kết quả hoặc mục trong measurements.md). Không có số → ghi rõ là phỏng đoán.
5. **Tách rõ "đo được" và "suy ra".** Ví dụ: "đo được KV cache chứa 36k token" khác "suy ra tối đa ~10 user RAG cùng lúc".
6. **Câu ngắn, mỗi đoạn một ý.** Bảng cho số liệu, chữ cho giải thích.
7. Chuyện không như dự kiến (lỗi, phải đổi cách làm) ghi thật, kể cả khi là do mình làm sai.

## Khung `bao-cao.md`

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
