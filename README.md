# kaglaw

Internal Kaggle research agent. Chat tự nhiên — agent dùng 33 tools để **tự viết
code notebook**, quản lý nhiều nick, push notebook, submit, đọc kết quả run, tổng
hợp số liệu và xuất Excel theo yêu cầu.

Có cả 2 mặt:
- **Chat UI (`/chat`)** — sai việc bằng tiếng Việt, agent tự gọi tools. Agent có thể
  **tạo file code Kaggle từ đầu** ngay trong chat (`create_notebook`), sửa code
  (`update_notebook_code`), push, rồi đọc lại log/score (`get_run_output`).
- **Dashboard** (Accounts / Notebooks / Runs / Live / Experiments / Batches / Competitions / Memory) — quản lý thủ công.
  Trang Notebooks: **sửa code ngay trong trình duyệt** (textarea + *Lưu local* / *Save & push* = tạo version
  mới trên Kaggle), nút *Test locally*, và **Import từ Kaggle** (`<user>/<slug>`) để kéo kernel có sẵn về sửa.

## Bộ nhớ & ngữ cảnh

- **Memory dài hạn** (`/memory`): fact/sở thích được **tự chèn vào MỌI cuộc chat** (vd "luôn dùng nick main,
  metric AUC, 5-fold"). Agent tự lưu qua `remember`, tra bằng `recall`, xoá bằng `forget`.
- **Nén hội thoại dài**: khi 1 chat quá dài (vượt `KAGLAW_CONTEXT_BUDGET_CHARS`), kaglaw tự **tóm tắt phần cũ**
  (rolling summary, cache trên `conversations`) và chỉ gửi phần gần nhất → không vỡ giới hạn token.
- Mỗi cuộc chat vẫn lưu **đầy đủ lịch sử** trong SQLite (mở lại còn nguyên).

## Vòng đời "tạo code → chạy → lấy kết quả"

1. `create_notebook(title, code, enable_gpu, competition_sources=[...])` — agent viết source
   (tách cell bằng `# %%`), kaglaw đóng gói thành `.ipynb` hợp lệ và đăng ký vào db.
2. `test_notebook_local(notebook_id)` — chạy thử ngay trên máy (subprocess + timeout) để bắt lỗi
   syntax/import/logic trước khi tốn quota GPU. Có nút *Test locally* ở trang Notebooks.
3. `push_notebook_to_accounts(notebook_id, nicks=[...])` — push lên 1..N nick (cần `confirm`).
4. `sync_run_status(run_id)` — cập nhật trạng thái; khi xong tự kéo output về `data/outputs/`.
5. `get_run_output(run_id, file="submission.csv")` — đọc log/score/submission để báo cáo.
6. `export_excel_report` — báo cáo Excel 8 sheet (Summary, Experiments, Notebooks, Runs,
   Submissions, GPU Usage, Budgets, Outputs & Logs).

## Experiment tracking (cho nghiên cứu)

Mỗi run là một **experiment**: lưu `params`, `metric` (CV/AUC/score tự parse từ log khi run
xong), `competition`, `tags`, `batch_id` (gom 1 lần push/sweep), và **snapshot code** đúng
version đã chạy vào `data/runs/<run_id>/` (tái lập được).

- Trang **`/experiments`**: bảng param × metric đã sort, tô đậm best, lọc theo comp/batch.
- Chat: `compare_experiments(competition=...)` để xem param nào cho score tốt nhất;
  `list_experiments`, `set_run_metric`, `reextract_metrics`.
- Quét tham số: `variant_push` lưu bộ param của từng biến thể → so sánh trực tiếp trong cùng batch.
- **CV vs LB**: `metric_value` = CV parse từ log (offline); `lb_public` = điểm leaderboard thật. kaglaw
  tự nối run↔submission khi sync (`autolink_run_scores`, hoặc `set_run_lb` chỉnh tay) — heuristic theo
  competition+nick+thời gian, nên kiểm lại nếu quan trọng.

## Parameter sweep & hàng đợi (Phase 2+3)

Chạy nhiều cấu hình tự động, phân bổ qua nhiều nick theo quota:

1. Viết notebook với placeholder `{{name}}` (vd `LR = {{lr}}`, `SEED = {{seed}}`).
2. `launch_sweep(notebook_id, grid={"lr":[0.1,0.05], "seed":[1,2]}, nicks=[...])` →
   bung thành N experiment, đưa vào **hàng đợi `jobs`**.
3. **Dispatcher** (scheduler, ~30s/lần) phóng job vào nick **còn slot trống + còn GPU budget**;
   retry lỗi tạm thời; gom theo `batch_id`.
4. Theo dõi: trang **`/batches`** (tiến độ + quota mỗi nick + best) hoặc `batch_status(batch_id)`;
   khi xong có **notification**. Hủy phần chưa chạy: `cancel_batch`.
5. Trang **`/status`** ("Live"): mỗi nick còn mấy slot + GPU còn lại + **đang chạy notebook/run nào** +
   mấy job đang chờ. Trong chat: tool `nick_status`.

Quota là **ước lượng** (Kaggle không có API): GPU hours = tổng runtime GPU 7 ngày gần nhất;
`MAX_CONCURRENT_PER_NICK`, `GPU_WEEKLY_HOURS`, `DISPATCH_INTERVAL` chỉnh qua env (xem config.py).

## Install

```powershell
cd D:\Claw_Kaggle   # thư mục dự án (package Python là `kaglaw`)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Add accounts

For each Kaggle account, drop its `kaggle.json` (download from
`kaggle.com/settings -> Create New Token`) into a sub-folder:

```
data\accounts\<nick>\kaggle.json
```

The `<nick>` folder name is the label used everywhere in the UI; the actual
Kaggle username is read from inside the JSON.

## Configure LLM

Vào http://127.0.0.1:8765/settings sau khi chạy server, chọn provider (Anthropic hoặc OpenAI)
và dán API key. Hoặc set biến môi trường trước khi chạy:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # cho Claude
$env:OPENAI_API_KEY    = "sk-..."       # cho OpenAI
```

## Run

```powershell
python run.py
```

Open http://127.0.0.1:8765 .

## Ví dụ lệnh chat

- "Viết cho tôi notebook baseline LightGBM cho comp `playground-series-s5e1`, bật GPU, đọc data từ /kaggle/input."
- "Sửa notebook #4: đổi n_estimators từ 500 thành 2000, thêm early stopping."
- "Push notebook #4 lên main rồi theo dõi, xong thì đọc log + score giúp tôi."
- "Tóm tắt tiến độ comp `playground-series-s5e1` qua các nick."
- "Tìm top 10 kernel public của comp X, sort theo votes."
- "Push notebook id 3 lên main, alt1, alt2 với version note 'lr=3e-4'."
- "Xuất Excel chỉ 2 sheet: best score mỗi nick + runs đang chạy."
- "Submit file `D:\subs\sub_v7.csv` lên titanic dưới nick alt2, message 'xgb tuned'."

## Layout

- `kaglaw/config.py` — paths / settings
- `kaglaw/db.py` — SQLite schema + helpers
- `kaglaw/accounts.py` — multi-account loader
- `kaglaw/kaggle_client.py` — `MultiKaggle` wrapper (thread-safe context switch)
- `kaglaw/actions.py` — high-level: push to N nicks, submit, sync
- `kaglaw/notebook_builder.py` — biến source text → `.ipynb`/`.py` và ngược lại (chat tự viết code)
- `kaglaw/scheduler.py` — APScheduler background poller
- `kaglaw/excel_export.py` — Excel report (8 sheets)
- `kaglaw/web/app.py` — FastAPI + HTMX UI
- `data/` — accounts, notebooks, pulled outputs, exports, sqlite db
