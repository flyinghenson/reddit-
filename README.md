# reddit_scene_crawler

根据**中文场景描述**自动生成英文检索词（可选），在 Reddit（非官方 JSON 端点）上爬取相关 **posts** 与 **comments+replies（合表）**，并以 **CSV** 方式落盘，支持**增量更新**与**断点续跑**。

## 输出文件

默认输出到 `exports/<scene_id>/`：

- `posts.csv`
- `comments.csv`（comments + replies 合在一起）
- `state.json`（断点续跑进度）
- `seen_posts.txt` / `seen_comments.txt`（增量去重）
- `todo_posts.jsonl`（待抓评论的 posts 队列）

## 安装

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果你本机 `pip install` 遇到证书问题（SSL），常见处理方式（按需选一种）：

- **企业网络/代理拦截证书**：配置 `HTTPS_PROXY/HTTP_PROXY` 或使用公司提供的 CA 证书。
- **临时跳过校验（不推荐，但可用于快速验证）**：

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## 运行（推荐：不依赖 LLM）

```bash
python3 scripts/crawl_reddit_scene.py \
  --scene "测试场景：白领下午犯困，想靠咖啡提神" \
  --keywords "afternoon slump coffee,midday crash caffeine" \
  --target-posts 200 \
  --max-pages-per-keyword 5 \
  --comment-sorts best,top \
  --expand-more
```

## 运行（使用 LLM 自动生成英文检索词）

需要环境变量：

- `OPENAI_API_KEY`
- `OPENAI_MODEL`（可选）
- `OPENAI_BASE_URL`（可选）

```bash
python3 scripts/crawl_reddit_scene.py \
  --scene "你的中文场景描述" \
  --keyword-count 30 \
  --target-posts 10000 \
  --max-pages-per-keyword 120 \
  --comment-sorts best,top \
  --expand-more
```

