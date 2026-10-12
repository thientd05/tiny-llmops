# Thiết kế thí nghiệm chạy tiếp được (resume)

> Quy tắc áp dụng cho **mọi thí nghiệm từ 2026-10-12** (CLAUDE.md quy tắc 20).
> Yêu cầu: thí nghiệm dừng giữa chừng vì bất kỳ lý do gì → chạy lại **cùng lệnh** → chỉ làm phần còn dở, không làm lại phần đã xong.

## Các tình huống dừng

| Tình huống | Máy thuê | Dữ liệu trên máy | Cách tiếp tục |
|---|---|---|---|
| Mất SSH, laptop tắt/ngủ | Vẫn chạy | Còn | Việc chạy trong `tmux` trên máy thuê nên không phụ thuộc SSH. Kết nối lại, xem tiến độ |
| Chủ động dừng để làm việc khác (> ~10 phút) / host mất máy / hết tiền | `destroy` (không dùng `stop`, xem vast-guide mục 1) | **Mất** | Thuê máy mới, đẩy thư mục kết quả từ local lên, chạy lại lệnh. Cài lại ~7 phút |
| Lỗi giữa chừng (server chết, hết bộ nhớ) | Vẫn chạy | Còn | Sửa nguyên nhân, chạy lại lệnh |

## Nguyên tắc

1. **Chia việc thành đơn vị nhỏ, độc lập.** Ví dụ một điểm đo = workload × số user × lần lặp, mất vài phút.
   Mỗi đơn vị có **ID tính từ cấu hình** (ví dụ `w2-c32-r2`), không tính từ thời gian chạy. Nhờ vậy chạy lại vẫn ra đúng ID cũ.
2. **Đánh dấu xong bằng file `done.json`, ghi cuối cùng.** Chỉ ghi sau khi mọi file kết quả của đơn vị đã ghi xong.
   Mọi file đều ghi kiểu nguyên tử: ghi ra `x.tmp` rồi đổi tên thành `x`. Đổi tên là thao tác một bước, nên không bao giờ có file chỉ ghi được một nửa.
3. **Chạy lại = cùng lệnh.** Script duyệt danh sách đơn vị và xử lý theo trạng thái:
   - có `done.json` hợp lệ → bỏ qua;
   - có thư mục nhưng không có `done.json` (đang chạy thì dừng) → xoá thư mục, chạy lại **riêng đơn vị đó** từ đầu;
   - chưa có thư mục → chạy.
4. **`done.json` chứa mã hash của cấu hình**: tham số đơn vị + sha256 workload + phiên bản phần mềm + lệnh khởi động server.
   Nếu cấu hình đã đổi thì hash khác, đơn vị sẽ được chạy lại thay vì bị bỏ qua nhầm (kết quả cũ không còn đúng cấu hình).
5. **Không có trạng thái ẩn giữa các đơn vị.** Mỗi đơn vị tự đưa server về trạng thái chuẩn trước khi đo (xoá prefix cache, warmup).
   Vì vậy dừng ở đâu, khởi động lại server bao nhiêu lần cũng không làm sai kết quả.
6. **Kéo kết quả về local liên tục.** Trong suốt phiên, ở local chạy vòng `rsync` mỗi ~2 phút. **Thư mục kết quả ở local là bản gốc.**
   Khi phải thuê máy mới: rsync thư mục kết quả local lên máy mới trước, rồi chạy lại lệnh.
7. **Setup chạy lại được nhiều lần.** Bước nào đã xong thì bỏ qua: venv đúng phiên bản vLLM thì không cài lại; model thiếu file thì tải tiếp, không tải lại từ đầu.
   Script khởi động server dùng lại server đang chạy nếu đúng cấu hình.
8. **Việc dài chạy trong `tmux` trên máy thuê**, không chạy trực tiếp trong phiên SSH.
9. **Mỗi đơn vị ghi lại máy và phiên chạy** (host_id, machine_id, instance, thời điểm khởi động server).
   Nếu phần tiếp theo chạy trên host khác: đo lại một điểm mốc (W2, 16 user) trên host mới để so với host cũ, và ghi vào báo cáo.
10. **Có lệnh xem tiến độ** (`--status`): số đơn vị đã xong / còn lại và ước tính giờ GPU còn cần.

## Kích thước đơn vị theo loại thí nghiệm

| Thí nghiệm | Đơn vị | Mất tối đa khi dừng |
|---|---|---|
| Sweep benchmark (P03, P04, tune GĐ1, GĐ2, GĐ5) | 1 điểm đo: workload × số user × lần lặp | 1 điểm đo (vài phút) |
| Phát lại trace (P05) | 1 mức tải (30 phút). Không cắt nhỏ hơn, vì hàng đợi request phải liên tục từ đầu đến cuối | ≤ 30 phút |
| Output tham chiếu (P06) | 1 nhóm 100 prompt | ~1 phút |
| lm-evaluation-harness (P06) | 1 tác vụ, bật `--use_cache` để lưu từng câu đã trả lời | vài câu |
| Build workload (P02) | Cả lần build (~2 phút CPU); file dataset tải dở thì tự tải tiếp | ~2 phút |
| Microbenchmark / profile kernel (GĐ3–4) | 1 kernel × 1 kích thước đầu vào | 1 lần đo |

## Quy trình một phiên thuê máy

```bash
# local: thuê máy, đẩy code + trạng thái kết quả lên
rsync -az results/<plan>/ root@<IP>:/root/results/<plan>/
# máy thuê, trong tmux: setup (bỏ qua phần đã xong) → server → chạy/chạy tiếp
tmux new -s exp 'bash setup_node.sh && bash serve_default.sh ... && python bench/sweep.py --config X --out /root/results/<plan>'
# local: kéo về liên tục trong suốt phiên
while true; do rsync -az root@<IP>:/root/results/<plan>/ results/<plan>/; sleep 120; done
# xem tiến độ
python bench/sweep.py --config X --out results/<plan> --status
```
