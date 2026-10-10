# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đỗ Đình Long
- MSSV: 2A202602673
- Email: long2110d@gmail.com
- Link repo (fork): https://github.com/long2110d/K4-L2L3-DAY23-DoDinhLong-2A202602673-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`):

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]` (199 frame), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.344, 0, 239, 2.523
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.267, 0, 239, 2.523
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  - Chỉ số rubric (hai mode như nhau): `precision_track = 502/(502+0) = 1.000`,
    `coverage = matches/det_tp = 502/519 = 0.967`; RMSE ≤ 0.45 m nên hệ số chất lượng q = 1.
    `rmse_fused − rmse_lidar = −0.0145 m` (≤ 0.05).
  - `matches`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks` **giống hệt** ở hai
    mode. Đây là hệ quả của thiết kế: camera chỉ chạy `ekf_update`, không cộng/trừ score, không
    khởi tạo hay xóa track (`TrackManager.manage_tracks` trả về ngay nếu sensor không phải lidar),
    nên tập track confirmed ở mỗi frame là như nhau; chỉ vị trí ước lượng khác.
  - Khác biệt duy nhất nằm ở sai số vị trí: `sum_sq_err` giảm từ 11.344 xuống 9.267 (≈ −18%), RMSE
    giảm 0.150 → 0.136 m. Đối chiếu từng frame trong `grade_run.log`: trong các frame có ghép,
    fused có `sum_sq_err` nhỏ hơn lidar ở 137 frame và lớn hơn ở 58 frame. Đo camera (u, v) thêm
    ràng buộc về hướng ngang/dọc nhìn từ camera FRONT, kéo track về gần tâm xe hơn.
  - 239 miss chủ yếu đến từ detector: 217 miss rơi vào các frame có `det_fn > 0` (tổng
    `det_fn = 222`, recall 0.70) — xe không được phát hiện thì không có đo LiDAR để giữ/ghép track.
    22 miss còn lại ở frame detector đủ, chủ yếu do độ trễ xác nhận: track mới cần score > 0.8
    (5 hit, `window = 6`) — track confirmed đầu tiên xuất hiện ở frame 4.
  - 0 ghost: với `fp = 16` của detector, track sinh từ false positive không đủ hit để confirmed
    và bị xóa khi score về 0.
  - Giới hạn: camera ở đây là tâm hộp 2D **ground-truth** FRONT cộng nhiễu seeded (σ = 5 px),
    không phải camera detector, nên mức giảm RMSE của fused là cận trên lạc quan; nó không
    chứng minh chất lượng một perception system camera thật, và chỉ có tác dụng với xe trong
    FOV camera FRONT.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   **Trả lời:** LiDAR cho `z = (x, y, z)` (m) trong hệ cảm biến, `R = diag(0.1², 0.1², 0.1²)`;
   `h(x) = R_vs·p + t` tuyến tính nên `H` là ma trận 3×6 hằng (khối xoay, cột vận tốc = 0).
   Camera cho `z = (u, v)` (pixel), `R = diag(σ_cam_i², σ_cam_j²) = diag(25, 25)` dựng trong
   `build_camera_measurement` ([camera_fusion.py](workspace/camera_fusion.py)); `h(x)` là
   phép chiếu pinhole phi tuyến `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s`
   (`camera_measurement_prediction`), nên EKF phải tuyến tính hoá bằng Jacobian 2×6 do platform
   tính tại trạng thái hiện tại. Camera không đo độ sâu: nó chỉ ràng buộc hướng nhìn, còn
   khoảng cách do LiDAR quyết định. Cùng một `ekf_update` ([kalman.py](workspace/kalman.py))
   phục vụ cả hai vì chỉ dùng `meas.z`, `meas.R`, `sensor.get_hx/get_H`.

2. Vì sao cần gating Mahalanobis trước khi gán?

   **Trả lời:** Khoảng cách Euclid bỏ qua độ bất định: 1 m có thể rất xa với track đã hội tụ
   nhưng gần với track mới có P lớn, và không so được m với pixel. `d² = γᵀS⁻¹γ` với
   `S = HPHᵀ + R` chuẩn hoá innovation theo bất định của cả track và sensor; nếu cặp đúng thì
   d² ~ χ² với bậc tự do = `dim_meas`. `chi2_gate` loại cặp có d² ≥ `chi2.ppf(0.995, dim)`
   (≈ 12.84 cho LiDAR 3D, 10.60 cho camera 2D), nên greedy không thể ghép một đo vô lý chỉ vì
   nó là cái gần nhất còn lại. Trong `association_cost_matrix`, kiểm tra `in_fov` còn đứng
   trước cả Mahalanobis để không chiếu điểm sau camera. Kết quả: 0 ghost trên cả 199 frame.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   **Trả lời:** Track-then-fuse theo nghĩa của lab: tracker duy trì một danh sách track (sinh từ
   LiDAR) và mỗi sensor được gán/cập nhật tuần tự vào cùng các track đó — mỗi frame: predict một
   lần → `associate_and_update(..., lidar)` → `associate_and_update(..., camera)`
   (`platform/fusion_lab/scripts/run_lab.py` dòng 200–209), không ghép đo thô hai sensor trước.
   Bằng chứng trên log: ở mọi frame, `confirmed`, `matches`, `ghosts`, `misses` của record
   `fused` trùng với record `lidar` (ví dụ frame 100: cả hai `confirmed=3, matches=3`), chỉ
   `sum_sq_err` khác (0.0601 → 0.0488). Tức camera không tạo/xóa track mà chỉ tinh chỉnh trạng
   thái của track có sẵn.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   **Trả lời:** Extrinsic/intrinsic sai làm `h(x)` lệch có hệ thống, nên innovation camera
   `γ = z − h(x)` không còn trung bình 0 mà có bias ổn định (ví dụ u luôn lệch một phía khi
   yaw extrinsic sai), và d² trung bình (NIS) vượt kỳ vọng `dim_meas = 2`. Hệ quả: nhiều cặp
   camera bị cổng χ² loại (camera gần như vô tác dụng), hoặc nếu lọt cổng thì update kéo track
   lệch ngang rồi lượt LiDAR kéo về — RMSE fused tăng, có thể vượt LiDAR-only. Kiểm tra: vẽ
   innovation camera theo thời gian của track confirmed, hoặc so NIS trung bình với 2.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   **Trả lời:** Khi `meas_list` rỗng thì không thể suy ra sensor từ `meas.sensor`, nhưng
   `manage_tracks` vẫn phải biết đây là lượt nào: lượt LiDAR rỗng nghĩa là mọi track trong FOV
   bị miss (giảm score, có thể xóa), còn lượt camera rỗng không được làm gì. Vì vậy
   `associate_and_update` luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas,
   sensor)` ở cuối ([association.py](workspace/association.py)), kể cả khi không có đo.
   LiDAR đo vị trí 3D đầy đủ, FOV rộng, nên đủ thông tin để khởi tạo state 6D và đánh giá tồn tại.
   Camera thiếu độ sâu, FOV hẹp (chỉ FRONT) và trong lab là nhãn mô phỏng: nếu camera được trừ
   score thì mọi xe ngoài FOV FRONT hay bị che sẽ bị xóa sai; nếu được sinh track thì không có
   độ sâu để khởi tạo. Nên camera chỉ gọi `ekf_update`.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   **Trả lời:** ([track_management.py](workspace/track_management.py)) Track mới: score =
   1/6, state `initialized`, P vị trí = R LiDAR xoay sang hệ xe, P vận tốc = diag(50², 50², 5²).
   Mỗi hit LiDAR: score += 1/6 (tối đa 1); miss LiDAR trong FOV: score −= 1/6. Hit khi chưa
   confirmed → `tentative`, và `confirmed` khi score > 0.8 (cần 5 hit). Track confirmed không
   bị hạ trạng thái: một miss từ 1.0 còn 5/6 ≈ 0.83 ≥ 0.6 nên vẫn sống. Xóa (OR) khi
   `P[0,0] > 9` hoặc `P[1,1] > 9` (bất định > 3 m), hoặc confirmed với score < 0.6, hoặc chưa
   confirmed với score ≤ 0. Chỉ lượt LiDAR đánh giá xóa.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Anthropic)
- Dùng cho phần nào (hàm, câu hỏi, debug): cài đặt Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`), dựng môi trường, chạy Waymo và soạn nháp báo cáo này từ số liệu của lần chạy.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` (128 passed, không failed/xfailed); chạy `fusion-run-lab --fusion compare --seed 0` frame 0–198; script đối chiếu `grade_run.log` với `metrics.json` (398 record, mỗi (mode, frame) một lần, `matches+ghosts==confirmed`, `matches+misses==valid_gt`); đối chiếu công thức EKF/Mahalanobis/pinhole với hướng dẫn kỹ thuật; tự đọc lại từng hàm.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
