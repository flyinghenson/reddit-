# 直连 Reddit 运行手册（可复制到另一台电脑给 AI 读）

本文档用于让另一台电脑/另一个 AI **不需要上下文**也能理解并执行本项目的任务：  
输入中文场景 → 用英文关键词在 Reddit 全站搜索 posts → 抓取 posts 的 comments + replies（合表）→ 输出 CSV，支持断点续跑与增量去重。  

## 背景与约束（务必先读）

- **本项目使用“直连 Reddit 非官方 JSON 端点”**（不使用官方 API key）。
- 直连时最常见问题是 **`403 Blocked`**（IP/网络环境被 Reddit 拦截）。一旦 403：
  - posts 搜索会变成 **0**
  - comments 拉取也会变成 **0**
  - 重跑/重启进程不会解决，必须换网络/IP 或降低被拦截概率
- **限流**：如果出现 `429`，程序会自动退避重试，但你仍应降低并发与请求频率。

> 备注：代码里也做了一个 PullPush（历史数据 API）兜底实现，但如果你只想“直连 Reddit”，请按本文档方式运行，并确保 **不启用 PullPush**。

## 代码位置与关键脚本

项目根目录：`/Users/hensonzhang/软件开发/数据分析/reddit_scene_crawler/`

- **单场景运行入口（用户每次只跑一个场景）**：`scripts/crawl_reddit_scene.py`
- **单场景预设入口（用户指定要跑哪个场景预设）**：`scripts/run_one_scene_preset.py`
- **爬虫实现**：`src/reddit_scene_crawler/reddit_crawler.py`
- **CSV/断点续跑存储**：`src/reddit_scene_crawler/scene_store.py`

## 输出结构（文件名带场景）

每个场景会输出到一个独立目录：

- `exports/<scene_id>__<场景名>/`

目录内会生成：

- `posts_<场景名>.csv`
- `comments_<场景名>.csv`（comments + replies 合表）
- `state.json`（断点续跑状态）
- `seen_posts.txt` / `seen_comments.txt`（去重集合）
- `todo_posts.jsonl`（待抓评论的 post 队列）
- `keywords.json`、`run_meta.json`

### CSV 格式说明（避免“换行导致一条记录多行”）

- `content` 字段里真实换行会被转换为字面量 `\\n`，保证 **CSV 一行 = 一条记录**。

## 环境准备

### Python

- 推荐 `Python >= 3.10`

### 依赖安装（可选）

脚本默认用标准库 `urllib` 发请求；安装 `httpx` 会更快更稳。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 直连 Reddit：运行前自检（最重要）

如果下面自检返回 403，**不要开始跑**，先换网络/IP。

```bash
python3 - <<'PY'
import urllib.request, urllib.parse
headers={'User-Agent':'Mozilla/5.0','Accept-Language':'en-US,en;q=0.9'}
url="https://old.reddit.com/search.json?"+urllib.parse.urlencode({'q':'productivity','sort':'relevance','t':'all','limit':3,'type':'link','raw_json':1})
req=urllib.request.Request(url,headers=headers)
try:
    with urllib.request.urlopen(req,timeout=20) as r:
        print("OK", getattr(r,'status',None))
except Exception as e:
    print("FAIL", type(e).__name__, e)
PY
```

预期：
- `OK 200`：可以开跑
- `FAIL ... 403 Blocked`：换网络/IP（手机热点、不同出口、不同 VPN 节点等）

## 直连 Reddit：单场景运行

### 关键参数

- `--scene`：中文场景描述（用于生成 scene_id & 写入元信息）
- `--scene-name`：用于**输出文件名标注场景**（强烈建议填）
- `--keywords`：英文关键词列表（逗号分隔）。推荐优先手工给关键词，避免依赖 LLM。
- `--target-posts 0`：不设 posts 上限（尽可能多）
- `--max-pages-per-keyword`：每个关键词最大翻页数（每页最多 100 条）
- `--expand-more`：尝试展开更多评论节点（评论更全，但更慢、更容易触发限流）
- `--max-comment-posts`：可选。默认 **0=不限制**（抓取所有已获取 posts 的 comments/replies）
- `--fresh`：清空该场景输出目录后重跑；不带 `--fresh` 则断点续跑

### 示例：抓取全量 comments（不限制，数据量可能很大）

```bash
cd "/Users/hensonzhang/软件开发/数据分析/reddit_scene_crawler"

# 直连 Reddit：确保不启用 PullPush
unset USE_PULLPUSH

MIN_REQUEST_INTERVAL=2.0 python3 -u scripts/crawl_reddit_scene.py \
  --scene "工作与生产力" \
  --scene-name "场景1_工作与生产力" \
  --fresh \
  --keywords "productivity,focus at work,procrastination,pomodoro,deep work" \
  --target-posts 0 \
  --max-pages-per-keyword 120 \
  --comment-sorts best,top \
  --expand-more
```

### 断点续跑

同一条命令去掉 `--fresh` 重新执行即可：

```bash
MIN_REQUEST_INTERVAL=2.0 python3 -u scripts/crawl_reddit_scene.py \
  --scene "工作与生产力" \
  --scene-name "场景1_工作与生产力" \
  --keywords "productivity,focus at work,procrastination,pomodoro,deep work" \
  --target-posts 0 \
  --max-pages-per-keyword 120 \
  --comment-sorts best,top \
  --expand-more
```

## 单场景预设运行（推荐：用户选择要跑的场景）

如果你希望“每次只跑一个场景，并且由用户指定要跑哪个场景”，用这个脚本最方便：  
它内置了 6 个场景的英文关键词预设，你只需要指定 `--preset`。

该预设脚本默认策略：
- **posts**：尽可能多（`--target-posts 0`）
- **comments**：不限制（默认抓取所有已获取 posts 的 comments/replies；数据量可能很大）
- **comments 深度**：开启 `--expand-more` 时尽可能展开评论树（更全但更慢）

```bash
cd "/Users/hensonzhang/软件开发/数据分析/reddit_scene_crawler"
unset USE_PULLPUSH

MIN_REQUEST_INTERVAL=2.0 python3 -u scripts/run_one_scene_preset.py \
  --preset "场景1_工作与生产力" \
  --target-posts 0 \
  --max-pages-per-keyword 120 \
  --expand-more \
  --fresh
```

可选 preset 列表：

- `场景1_工作与生产力`
- `场景2_运动与体能训练`
- `场景3_睡眠与恢复`
- `场景4_心理健康与情绪`
- `场景5_特殊职业_夜班工作`
- `场景6_慢性病或健康问题管理`

## 常见问题排查

### 1) 全是 0 条 posts/comments

- 最常见：**403 Blocked**
  - 先跑“运行前自检”
  - 换网络/IP 后再跑

### 2) 429 频繁

- 降低频率：把 `MIN_REQUEST_INTERVAL` 调大（例如 3~10）
- 不要并发跑多个场景（每次只跑一个场景更稳）
- 先关掉 `--expand-more`（评论树展开最耗请求）

### 3) 输出 CSV 看起来“断行/格式乱”

- 本项目已经把正文换行转换成了 `\\n`，理论上不会再“断行”。
- 如果你用的工具解析 CSV 仍有问题，建议用 Python `csv` 模块或 pandas 读取验证。

