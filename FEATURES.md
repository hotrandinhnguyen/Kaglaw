# 🦀 kaglaw — Tài liệu chức năng

> **kaglaw** = một agent chat + dashboard điều khiển **nhiều tài khoản Kaggle**: tự viết code,
> test, chạy hàng loạt theo quota, thu thập & so sánh kết quả, và xuất Excel.
> Tool nội bộ, điều khiển bằng tiếng Việt tự nhiên.

- **Stack:** Python 3.12 · FastAPI + Jinja2 + HTMX · SQLite (WAL) · APScheduler · openpyxl · kaggle SDK · Anthropic/OpenAI SDK
- **Package:** `kaglaw` · **Thư mục:** `D:\Claw_Kaggle` · **Chạy:** `python run.py` → http://127.0.0.1:8765
- **Số tool agent:** 54 · **Sheet Excel:** 8 · **Trang web:** 10

---

## Mục lục
1. [Vòng đời nghiên cứu (end-to-end)](#1-vòng-đời-nghiên-cứu-end-to-end)
2. [Quản lý nhiều nick](#2-quản-lý-nhiều-nick-kaggle)
3. [Viết & quản lý code qua chat](#3-viết--quản-lý-code-notebook-qua-chat)
4. [Test code local trước khi push](#4-test-code-local-trước-khi-push)
5. [Push & cơ chế version của Kaggle](#5-push--cơ-chế-version-của-kaggle)
6. [Chạy hàng loạt — parameter sweep](#6-chạy-hàng-loạt--parameter-sweep)
7. [Hàng đợi + quota (dispatcher)](#7-hàng-đợi--quota-dispatcher)
8. [Live status theo nick](#8-live-status-theo-nick)
9. [Lấy kết quả & trích metric](#9-lấy-kết-quả--trích-metric)
10. [Experiment tracking & so sánh](#10-experiment-tracking--so-sánh)
11. [Submission & điểm leaderboard](#11-submission--điểm-leaderboard)
12. [Nghiên cứu trên Kaggle](#12-nghiên-cứu-trên-kaggle)
13. [Thống kê & Excel (8 sheet)](#13-thống-kê--excel-8-sheet)
14. [Bộ nhớ & ngữ cảnh](#14-bộ-nhớ--ngữ-cảnh)
15. [Tự động hoá nền + thông báo](#15-tự-động-hoá-nền--thông-báo)
16. [Giao diện & an toàn](#16-giao-diện--an-toàn)
17. [Danh sách 54 tool](#17-danh-sách-54-tool)
18. [Cấu hình (env) & dữ liệu](#18-cấu-hình-env--dữ-liệu)
19. [Giới hạn đã biết](#19-giới-hạn-đã-biết)

---

## 1. Vòng đời nghiên cứu (end-to-end)

```
Viết code (chat) ─► Test local ─► Push/Sweep qua nhiều nick (theo quota)
      ▲                                          │
      │                                          ▼
   Sửa code ◄── So sánh /experiments ◄── Tự parse CV + nối điểm LB ◄── Run xong
                                          │
                                          ▼
                            Xuất Excel 8 sheet · Notification khi batch xong
```

**Ví dụ một câu lệnh:** *"Viết notebook LGBM cho comp `playground-s5e1` với `N_EST={{n_est}}`, `LR={{lr}}`,
test local rồi sweep grid n_est=[500,1000], lr=[0.05,0.1] qua nick main+alt1."*

---

## 2. Quản lý nhiều nick Kaggle
- Mỗi nick = 1 thư mục `data/accounts/<nick>/kaggle.json`. Thêm/xoá/rescan qua UI `/accounts` hoặc chat.
- Wrapper thread-safe tự đổi credential đúng nick cho mọi lời gọi Kaggle API (`as_account`, có khóa toàn cục).
- Tool: `list_accounts`, `remove_account`.

## 3. Viết & quản lý code notebook (qua chat)
- **Tạo mới:** agent tự viết source → đóng gói `.ipynb`/`.py` hợp lệ + đăng ký vào db. Tách cell bằng `# %%`, markdown bằng `# %% [markdown]`.
- **Đọc / sửa:** đọc lại dạng `# %%`, sửa toàn bộ hoặc vá từng dòng (`replacements`), giữ `.bak`.
- **Sửa ngay trong trình duyệt:** trang Notebooks → *View / edit code* (textarea + nút **Lưu local** / **Save & push**).
- **Import từ Kaggle:** kéo kernel có sẵn (`<user>/<slug>`) về thành notebook để sửa rồi push version mới.
- Tool: `create_notebook`, `get_notebook_code`, `update_notebook_code`, `import_kernel`, `register_notebook`, `delete_notebook`.

## 4. Test code local trước khi push
- Chạy thử notebook ngay trên máy (subprocess + timeout) để **bắt lỗi syntax/import/logic** trước khi tốn quota GPU.
- Parse metric từ stdout, liệt kê file output (vd `submission.csv`), từ chối notebook còn `{{placeholder}}`.
- UI: nút *Test locally* ở trang Notebooks. Tool: `test_notebook_local`.

## 5. Push & cơ chế version của Kaggle
- Mỗi lần **push** = đúng thao tác Kaggle **"Save Version → Save & Run All (Commit)"**: tạo **version mới + chạy** trong batch, sinh output.
- `version_notes` = message của version; `version_number` được lưu vào bảng `runs`.
- **Quan trọng:** sửa code trong kaglaw chỉ đổi **file local**; phải **push** mới lên Kaggle. Kaggle API chỉ pull/push (như git) — không có "sửa cell trực tiếp trên server".
- Nút **Save & push** trong editor = lưu + push 1 version mới dưới các nick chọn.
- Tool: `push_notebook_to_accounts`, `variant_push`.

## 6. Chạy hàng loạt — parameter sweep
- Notebook đặt placeholder `{{lr}}`, `{{seed}}`… → `launch_sweep(grid={...})` bung thành N thí nghiệm.
- `search="grid"` (mọi tổ hợp) hoặc `search="random", n=20`. Tự gắn `batch_id`, snapshot code mỗi run.
- Tool: `launch_sweep` (cần confirm), `variant_push`.

## 7. Hàng đợi + quota (dispatcher)
- Sweep đưa job vào **hàng đợi `jobs`**; **dispatcher** (scheduler ~30s/lần) phóng job vào nick **còn slot + còn GPU budget**, retry lỗi tạm thời.
- Tôn trọng `MAX_CONCURRENT_PER_NICK`. Theo dõi/điều khiển: `batch_status`, `list_batches`, `cancel_batch`, `dispatch_jobs_now`; trang `/batches`.

## 8. Live status theo nick
- Trang **`/status` ("Live")** + tool `nick_status`: mỗi nick hiện **đang chạy run/notebook nào**, còn mấy slot, GPU còn lại, mấy job đang chờ.
- Quota tổng quan: `list_budgets`, `nick_budget`.

## 9. Lấy kết quả & trích metric
- Run xong tự kéo output về `data/outputs/`; `get_run_output` đọc log/submission/score.
- **Tự parse metric** (CV/AUC/RMSE/score…) từ log bằng regex cấu hình được → cột số để sort/so sánh.
- Tool: `get_run_output`, `sync_run_status`, `sync_all_active_runs`, `set_run_metric`, `reextract_metrics`.

## 10. Experiment tracking & so sánh
- Mỗi run = **experiment**: `params` + `metric` + `competition` + `tags` + `batch_id` + **snapshot code đúng version** (tái lập được).
- `compare_experiments` → ma trận **param × metric** đã sort + `best`. Trang **`/experiments`**: bảng + **biểu đồ SVG** (best tô xanh) + lọc theo comp/batch.
- Tool: `list_experiments`, `compare_experiments`.

## 11. Submission & điểm leaderboard
- Submit CSV dưới nick bất kỳ; sync submissions + leaderboard → tính **rank theo public score**.
- **CV vs LB:** `metric_value` = CV parse từ log (offline); `lb_public` = điểm leaderboard thật. kaglaw **tự nối run↔submission** (heuristic theo comp+nick+thời gian) khi sync.
- Tool: `submit_to_competition`, `sync_submissions_for_competition`, `autolink_run_scores`, `set_run_lb`, `top_scores_per_nick`, `summarize_competition_progress`.

## 12. Nghiên cứu trên Kaggle
- Search competition / kernel / dataset; xem mô tả + deadline + **file train/test**; **tải source notebook public** về học approach; tra web (DuckDuckGo).
- Tool: `kaggle_search_competitions`, `kaggle_search_kernels`, `kaggle_search_datasets`, `kaggle_get_leaderboard`, `kaggle_competition_info`, `kaggle_competition_data_files`, `kaggle_get_kernel_source`, `web_search`, `read_local_file`, `list_local_dir`.

## 13. Thống kê & Excel (8 sheet)
`export_excel_report` (hoặc `/export.xlsx`):

| # | Sheet | Nội dung |
|---|-------|----------|
| 1 | **Summary** | Best public/private score per (competition, nick) + #subs + best rank |
| 2 | **Experiments** | 1 dòng/run: mỗi param 1 cột + metric (cho nghiên cứu) |
| 3 | **Notebooks** | Notebook đã đăng ký (id, title, flags, #runs, path) |
| 4 | **Runs** | Nick + version + status + runtime |
| 5 | **Submissions** | Competition + score + LB rank |
| 6 | **GPU Usage** | Tổng runtime/nick (ước lượng giờ GPU) |
| 7 | **Budgets** | Slot trống + GPU còn lại/nick (ước lượng) |
| 8 | **Outputs & Logs** | Output dir + log summary mỗi run |

- Excel tùy biến theo SQL: `build_custom_excel`. Query tự do (chỉ SELECT): `db_query`.

## 14. Bộ nhớ & ngữ cảnh
- **Memory dài hạn (`/memory`):** fact/sở thích được **tự chèn vào MỌI cuộc chat** (vd "luôn dùng nick main, metric AUC, 5-fold"). Tool: `remember`, `recall`, `forget`, `update_memory`.
- **Nén hội thoại dài:** chat quá dài → tự **tóm tắt phần cũ** (rolling summary, cache trên `conversations`) + chỉ gửi phần gần nhất → không vỡ giới hạn token.
- Mỗi chat vẫn lưu **đầy đủ lịch sử** trong SQLite (mở lại còn nguyên).

## 15. Tự động hoá nền + thông báo
- Scheduler tự: sync trạng thái run, sync submissions, **phóng job hàng đợi**, **báo notification khi batch xong**.
- Tool: `list_notifications`.

## 16. Giao diện & an toàn
- **Chat `/chat`:** streaming SSE, gọi 54 tool, render markdown/bảng. Hỗ trợ Anthropic Claude **hoặc** OpenAI (chọn provider/model/key ở `/settings`).
- **Dashboard:** Accounts · Notebooks · Runs · Live · Experiments · Batches · Competitions · Memory · Settings.
- **An toàn:** thao tác nguy hiểm (push/submit/sweep/xoá) bắt buộc **confirm preview** trước khi thực thi.

---

## 17. Danh sách 54 tool

> **D** = destructive (cần `confirm: true`); còn lại auto-run.

### Đọc / liệt kê nội bộ
`list_accounts` · `list_notebooks` · `list_runs` · `list_submissions` · `list_tracked_competitions` · `top_scores_per_nick` · `db_query`

### Viết & quản lý code
`create_notebook` · `get_notebook_code` · `update_notebook_code` · `test_notebook_local` · `import_kernel` · **D** `register_notebook` · **D** `delete_notebook`

### Push & chạy hàng loạt
**D** `push_notebook_to_accounts` · **D** `variant_push` · **D** `launch_sweep` · `dispatch_jobs_now` · `batch_status` · `list_batches` · `cancel_batch`

### Kết quả & experiment
`get_run_output` · `sync_run_status` · `sync_all_active_runs` · `set_run_metric` · `reextract_metrics` · `list_experiments` · `compare_experiments`

### Submission & leaderboard
**D** `submit_to_competition` · `sync_submissions_for_competition` · `autolink_run_scores` · `set_run_lb` · `summarize_competition_progress`

### Quota & live status
`nick_status` · `nick_budget` · `list_budgets` · `list_notifications`

### Nghiên cứu Kaggle / web / file
`kaggle_search_competitions` · `kaggle_search_kernels` · `kaggle_search_datasets` · `kaggle_get_leaderboard` · `kaggle_competition_info` · `kaggle_competition_data_files` · `kaggle_get_kernel_source` · `web_search` · `read_local_file` · `list_local_dir`

### Excel
`export_excel_report` · `build_custom_excel`

### Bộ nhớ dài hạn
`remember` · `recall` · `forget` · `update_memory`

### Quản trị tài khoản
**D** `remove_account`

---

## 18. Cấu hình (env) & dữ liệu

**Biến môi trường chính** (xem `kaglaw/config.py`):

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `KAGLAW_MAX_CONCURRENT_PER_NICK` | 1 | Số kernel chạy song song tối đa/nick |
| `KAGLAW_GPU_WEEKLY_HOURS` | 30 | Quota GPU tuần (ước lượng) |
| `KAGLAW_DISPATCH_INTERVAL` | 30 | Chu kỳ dispatcher (giây) |
| `KAGLAW_MAX_SWEEP_JOBS` | 200 | Trần số job mỗi sweep |
| `KAGLAW_LOCAL_RUN_TIMEOUT` | 300 | Timeout test local (giây) |
| `KAGLAW_CONTEXT_BUDGET_CHARS` | 48000 | Ngưỡng nén hội thoại |
| `KAGLAW_CONTEXT_RECENT_CHARS` | 20000 | Phần gần nhất giữ nguyên khi nén |
| `KAGLAW_MEMORY_MAX_CHARS` | 8000 | Trần memory chèn vào prompt |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | Khóa LLM (hoặc nhập ở `/settings`) |

**Thư mục dữ liệu** (`data/`): `accounts/` · `notebooks/` · `runs/` (snapshot code) · `outputs/` · `local_runs/` · `exports/` · `kaglaw.sqlite3`

**Bảng SQLite:** `accounts`, `notebooks`, `runs` (= experiment), `submissions`, `jobs`, `notifications`, `memories`, `conversations`, `messages`, `settings`.

---

## 19. Giới hạn đã biết
- **GPU quota là ước lượng** — Kaggle không có API quota; tính từ tổng runtime GPU 7 ngày.
- **Autolink LB là heuristic** theo thời gian — kiểm lại nếu cần chính xác tuyệt đối.
- **Sandbox local không cô lập mạnh** — chạy code tin tưởng (do agent/bạn viết); thư viện phải cài sẵn, đường dẫn `/kaggle/input` có thể không có.
- **Không sửa trực tiếp ô code trên server Kaggle** — theo thiết kế API (pull/push như git).

---

*Tạo tự động bởi kaglaw docs — cập nhật khi thêm chức năng.*
