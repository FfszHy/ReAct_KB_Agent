# ReAct_KB_Agent（pkb-agent）项目技术总结

> 用途：撰写简历中「个人项目 / 项目经历」部分的事实来源。所有数据均来自本仓库代码与已冻结的评测产物，可直接引用；标注「实测」的指标来自 `data/evals/fastapi-0.115/results/` 下的真实运行记录。

---

## 一、项目定位（一句话）

一个**以评测驱动的、可验证回答的个人知识库 Agent 系统**：基于 Supabase（Postgres + pgvector）构建混合检索知识库，用 DeepSeek 原生 Function Calling 驱动 ReAct 循环，Agent 运行时**只能通过带权限与全链路追踪的工具层访问外部世界**，并强制输出机器可校验的「答案 + 原子论断 + 引用」JSON 契约，无法被证据支撑时返回显式拒答而非编造。

**关键词一句话版**：ReAct Agent × 混合检索 RAG × 可验证回答契约 × 动态工具权限 × 可复现评测体系。

---

## 二、要解决的问题（简历背景段可用）

普通 RAG Demo 的典型缺陷，本项目的每一项设计都在针对性解决：

| 常见问题 | 本项目的处理 |
|---|---|
| Agent 幻觉引用、编造来源 | 运行时内存证据台账（evidence ledger）+ JSON 引用契约，伪造/过期/未检索到的 citation id 直接拒绝 |
| 检索只做向量、召回不稳 | pgvector 语义检索 + Postgres 全文检索混合，两套融合策略（RRF / 归一化加权）可切换并做消融 |
| 工具随意调用、越权抓取 | 三层权限（allow/ask/deny）+ 参数级约束 + SSRF 防护 + 动态 DB 覆盖策略 |
| 代码「看起来能跑」但没有数 | 5 策略检索消融 + 41 篇固定语料 + 120 题冻结测试集 + bootstrap 置信区间 + 逐题 win/loss/tie |
| 评测指标自说自话 | 自动指标与人工审计指标严格分离，未人工复核的语义指标留空，不把启发式包装成质量结论 |
| 可观测性写失败拖垮主流程 | Trace 有界重试 + 非致命降级，写失败记为「可观测性降级」而不是 Agent 执行失败 |

---

## 三、整体架构

```
CLI (Typer+Rich) / Next.js 工作台
                │  (HTTP + SSE)
                ▼
        FastAPI 边界层  ────────────────────────────┐
                │                                  │
                ▼                                  ▼
       DeepSeek ReAct Runtime              审批代理 ApprovalBroker
       (Thought → Action → Observation)    (web_fetch / memory_write
                │                            需浏览器显式批准)
                ▼
   Tool Registry + Permission Manager
   ┌─────────────────────────────────────────────────┐
   │ rag_search  rag_read  rag_list_documents        │  ← 知识库检索
   │ web_search  web_fetch                           │  ← 外部网络（受控）
   │ memory_search  memory_write                     │  ← 跨轮记忆
   │ calculator  now                                 │  ← 确定性工具
   └─────────────────────────────────────────────────┘
                │
                ▼
        Repositories (仓储层)
                │
                ▼
   Supabase Postgres：documents / document_chunks /
   chunk_embeddings / agent_runs / agent_steps /
   tool_calls / task_memory / tool_permissions
```

**核心不变量（面试可重点讲）**：Agent Runtime **从不**直接触碰 Supabase、网络或记忆系统，它只能调用工具。每一次工具调用都强制经过固定流水线：

```
参数校验 → 权限检查 → 追踪记录 → 执行 → 结果截断 → 异常包装
```

这条约束让「Agent 的行为边界」是代码级可审计的，而不是靠提示词自觉。

**代码规模**：`src/pkb_agent` 约 10,800 行 Python（13 个功能子模块），测试 25 个文件约 4,300 行；数据库迁移 9 个 SQL 文件；前端 Next.js 工作台 8 个组件。

---

## 四、核心实现原理

### 4.1 ReAct 循环（`agent/runtime.py`）

- **推理-行动循环**：每轮调用 DeepSeek（原生 `tools` 参数）→ 若返回 `tool_calls` 则执行工具、把 Observation 追加回消息；若返回纯文本则进入「最终答案校验」分支。
- **步数预算与收尾保留**：`max_steps=20`，其中预留 3 步作为「无工具收尾预算」，实际工具预算为 17 步。**预算耗尽不是抛 max-step 错误**，而是注入一条收尾指令，强制模型在不再调用工具的前提下输出最终 JSON——避免把「反复检索」变成运行失败。
- **单轮多工具调用**：一次模型补全可能返回多个 tool_call，循环按其批内位置逐个执行，并在预算边界内补齐「对应每个 tool_call 的 tool 消息」，保证提供方消息协议合法（不会遗留下未回应的调用导致下一轮被拒）。
- **异常兜底**：若提供方在 `tools=None` 时仍返回 tool_call，运行时仍会补 tool 响应并重试，最终退化为标准的「证据不足」拒答而非基础设施错误。

### 4.2 可验证回答契约（`agent/verification.py`）

最终模型输出被约束为**机器校验的 JSON 对象**，而非自由文本：

```json
{
  "status": "grounded | insufficient_evidence",
  "answer": "面向用户的完整解释",
  "claims": [{"text": "...", "kind": "fact|inference", "citations": ["kb:..."]}],
  "citations": [{"id": "kb:..."}]
}
```

校验规则（任一不满足即判定非法并触发修复）：
- 根字段白名单，未知字段直接拒绝；`answer` 非空且 ≤ 20,000 字符；`claims` ≤ 100；`citations` ≤ 100。
- `citations[i]` **只允许含 `id` 字段**——用户看到的引用元数据必须由运行时提供，模型只能「选择」证据，不能「编造」证据元数据。
- citation id **必须存在于本轮运行的内存证据台账**；不存在（伪造、陈旧、跨轮）即拒绝。
- `web_page` 类型的引用若已过期（超过 `evidence_ttl_hours=168h`）即拒绝。
- `grounded` 必须至少 1 条 claim 且至少 1 条 citation；`insufficient_evidence` **禁止**携带 claims/citations（拒答必须写在 `answer` 里）。
- 每条 claim 的 `kind` 只能是 `fact`（来源直接陈述）或 `inference`（模型基于被引事实的推断），两者都强制带引用。
- 所有 citation 必须被至少一条 claim 引用，孤立的「挂着不用」的引用被判非法。

**修复与拒答路径**：
- 校验失败 → 注入修复指令（列出全部校验错误 + 当前允许引用的证据清单）→ 重试（默认上限 2 次），模型可在修复轮继续检索补证。
- 仍失败 → 返回确定性的 `insufficient_evidence` 拒答载荷，**绝不把非法文本直接透传**。
- CLI 用不同面板分别渲染「原文事实 / Source facts」与「模型推断 / Model inferences」；`--json` 暴露完整答案对象、证据与校验审计。

**证据台账的构建**：`extract_evidence()` 从**成功**的工具观测中生成引用合格记录——
- `rag_search` / `rag_read` → `kb:{chunk_id}`（含标题、定位符、excerpt、doc_id/chunk_index/score 等元数据）
- `rag_list_documents` → 每次目录页折叠成**一条** `kbcatalog:{offset}` 级证据（避免为每份资料重复一条 UUID 引用）
- `web_fetch` → `web:{sha256(url|fetched_at)[:24]}`，携带抓取时间、域名、信任等级、过期状态、内容 SHA-256、截断状态
- **关键设计**：`web_search` 的结果摘要**永远不具备引用资格**。必须先用 `web_fetch` 把页面取回来才可作为证据——从机制上区分「听说」和「读到」。

### 4.3 混合检索（`rag/`）

**两路召回并行**（`asyncio.gather` + `asyncio.to_thread` 复用同步 Supabase 客户端）：
- **向量路**：pgvector 余弦距离 `<=> `，SQL 函数 `rag_vector_search`，返回 `1 - distance` 作为相似度。
- **词法路**：Postgres 全文检索 `tsvector @@ websearch_to_tsquery('simple', q)`，`ts_rank` 打分，SQL 函数 `rag_fts_search`。
- 候选池为 `top_k × 3`，检索在数据库函数内完成（Supabase 官方推荐的 RPC 模式）。

**FTS 查询投影**（一个容易被忽略但很实用的细节）：用户提问直接喂给 `websearch_to_tsquery` 会被当作「相邻词 AND」，只要一个虚词不在原文就零召回。因此先把自然语言问题投影为**宽容的 OR 查询**：正则抽词 → 去停用词表 → 去重 → 上限 10 个词 → `OR` 连接；非拉丁输入（如中文）回退原查询，交给语言适配的 text-search 配置。

**两种融合策略**（这是消融实验的核心之一）：
1. **RRF（Reciprocal Rank Fusion，纯排名融合）**：`score = Σ w_i / (k + rank_i)`，`k=60`，`vector_weight=0.6 / fts_weight=0.4`。只依赖排名，天然免归一化。
2. **归一化加权分数融合**：对两路原始分数（余弦相似度与 `ts_rank` 量纲完全不同）**各自独立做 min-max 归一化**，再按 0.6/0.4 加权合并。

**工程决策**：生产 Agent 的 `rag_search` 用**归一化加权混合**；RRF 作为检索消融的对照策略保留。这个「谁做生产默认值」不是拍脑袋，而是由下面的冻结测试集数据决定的（见 §6.3）。

**其他细节**：
- 查询重写产出多个 query 时，工具并发跑全部 query，同一 chunk 只保留最高分那次（`_merge_hits`）。
- Chunking：按自然边界切分（空行、以及包含中文 `。！？` 的句末），用 `tiktoken cl100k_base` 做 token 感知的尺寸控制，`chunk_size=800 tokens`，`chunk_overlap=120`，并保留 char_start/char_end 便于溯源。
- 入库：内容 SHA-256 去重；只对「缺少当前模型 + 当前维度向量」的 chunk 补嵌（换模型后重跑不会重复计费/混入异构向量）；embedding 以 `on_conflict=chunk_id` upsert 覆盖；PDF 用 pypdf 抽取（支持文件与内存字节流两种入口）。

### 4.4 工具层与动态权限（`tools/`）

**工具抽象**：`BaseTool` 用声明式 `ToolParam` 描述参数，自动生成 DeepSeek 的 JSON Schema（`to_schema`），并做基础类型校验（含 `bool` 不是 `integer/number` 的边界处理）。

**依赖注入**：每个工具拿到统一的 `ToolContext`（settings / supabase / trace / repositories / services / run_id / user_id / confirm_callback）。工具**不 import 全局单例**，运行时装配上下文再注入——保证可测试性，同时守住「Runtime 不碰数据库」的不变量。

**三层权限 + 参数级约束**：
- 基线策略在 `config/permissions.yaml`，三种档位：`allow` / `ask`（需交互确认）/ `deny`。
- 参数级约束在 `constraints`：`max_top_k`、`max_results`、`max_fetch_chars`、`max_content_chars`、`allowed_schemes` 等，由 PermissionManager 在执行前校验。
- **动态覆盖**：全局 `tool_permissions` 表的一行**整条替换**该工具的 YAML 规则（permission + constraints，不做局部合并，保证运行时行为可预测）。
- **刷新策略**：默认 `refresh_seconds=0`，即每次工具调用前刷新——长期存活的运行时能立刻观察到策略编辑；可调大以换取更少 DB 读。
- **故障回退**：DB 覆盖读取失败时**清空动态规则并回退 YAML 基线**（绝不能留下过期的动态 allow/deny 生效），同时在事件流里发出 `permission_overrides_error`。
- **非交互安全默认**：`ask` 类工具在非交互模式下**默认拒绝**；同步路径若拿到协程会 close 掉并判为拒绝（不会误当作 True）。

**SSRF / URL 安全**（`security/url_safety.py`）：仅允许 http/https；拒绝 netloc 中的 userinfo；阻断私网/环回/链路本地/组播/保留/未指定 IP（`ipaddress` 判定）；阻断 `.local` / `.internal` 后缀；内置云元数据端点黑名单（`169.254.169.254`、`metadata.google.internal`、`localhost` 等）。

### 4.5 查询重写（`prompts/`）

- 提示词正文在 `config/prompts/*.md`，其**生命周期在 `manifest.yaml` 里显式声明**，而不是靠文件名猜：`system_react` 每轮包含一次；`memory_policy` 仅在 `memory_write` 注册时才注入；`query_rewrite` 仅在紧邻 `rag_search` / `web_search` 前运行。
- 重写行为：`temperature=0`、`max_tokens=240`、要求 JSON 输出、最多 3 个聚焦 query；**失败即回退原 query，绝不阻塞检索工具**；按 `(tool, query, question)` 缓存（缓存命中不重复计费）。
- 溯源：trace 同时记录 **prompt hash + 原始参数 + 生效参数**（需迁移 `008_prompt_trace.sql`）。

### 4.6 记忆（`memory/`）

- 策略层白名单：`kind ∈ {preference, fact, decision, reference, note}`，`scope ∈ {short, long}`，内容上限 4000 字符，**疑似密钥内容在落库前被拒绝**。
- `short` = 单次运行内有效，`long` = 跨轮持久。

### 4.7 可观测性（`trace/`）

- **密钥脱敏**：工具参数、原始参数、prompt 上下文、观测结果、错误文本在写库前统一脱敏。
- **有界重试**：仅对**瞬时性**传输故障（超时/连接重置/连接拒绝/网关错误等，含被包裹的 `__cause__` / `__context__`）做最多 3 次指数退避重试；仓储写入是幂等的（稳定 ID + upsert），重试安全。
- **非致命降级**：重试耗尽后返回 `False`，运行时累加 `trace_write_failure_count` 并**继续作答**——可观测性写慢不能丢掉一个本来可用的答案；评测产物把它单独报告为「可观测性降级」。
- **成本核算**：基于已完成 usage 计算（非事前预估）。读取 DeepSeek 的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 分别计价（¥0.02 / ¥1.00 / ¥2.00 每百万 token，CNY），兼容端点未返回该字段时保守按 cache-miss 计价。

---

## 五、可视化工作台（第二形态）

- **后端**：FastAPI + **SSE** 边界层。`RunSession` 维护内存重放日志（带单调 sequence、上限 500 条）+ 多订阅者 fan-out；SSE 带 `retry`、15s keep-alive 心跳、断连检测；事件含 `start / plan / tool_call / tool_result / answer / answer_verification_failed / run_finished` 等类型。
- **人工审批闭环**：`ApprovalBroker` 让 `ask` 类工具（`web_fetch` / `memory_write`）在浏览器中显式批准，超时 300s；未批准则记 `ToolPermissionDenied` 并安全拒答。这是「人类在环（human-in-the-loop）」的落地形式，而不是提示词里写一句"请谨慎"。
- **接口**：文档上传（≤25MB，PDF/txt）→ 入库 → 提问 → SSE 订阅运行 → 引用卡片可点击跳回原始 chunk → 展开检索候选 / 重放 Trace 时间线 / 审批受保护工具 / 查看端到端耗时、token 用量、成本与工具成功率。
- **前端**：Next.js 工作台，把「证据优先」的一条链路做透（AnswerPanel / RetrievalPanel / SourceInspector / TraceTimeline / ApprovalDialog / RunMetrics / KnowledgeSidebar / RunDetails）。

---

## 六、评测体系（本项目最大的差异化亮点）

### 6.1 基准数据集

| 基准 | 语料 | 领域 | 开发集 | 冻结测试集 |
|---|---|---|---|---|
| **tech-multidomain-v1.1**（主基准） | 41 篇 | FastAPI `0.115.0` / Pydantic `v2.10.6` / Kubernetes `release-1.31` / SQLAlchemy `rel_2_0_36` | 80 题（50 + 30） | **120 题（88 可答 + 32 拒答）** |
| fastapi-0.115（回归基线） | 10 篇 | FastAPI `0.115.0` | 50 题 | 30 题（22 可答 + 8 拒答） |

> 基准版本管理是刻意设计的：v1.1 通过 `benchmark.json` 的 `split_includes` 在加载时把 v1.0 的跨域用例**组合**进来（50 本地 FastAPI 题 + 30 来自 v1.0 = 80 题 dev；30 + 90 = 120 题 test），并同步修复了一处来源标注缺陷（`test-query-02` 改引真正记录了 `Query(alias=...)` 的固定版本页面）。v1.0 原样保留供历史对比——**不发生静默重写**。

**刻意分层设计的挑战覆盖**（不是一坨单文档事实题）：15 题多文档、12 题改写/同义、18 题术语歧义、13 题**版本陷阱**（必须答对应该版本的行为）、32 题**要求诚实拒答**、8 题**受保护的外部抓取**（评测中确认一律拒绝，期望行为是"选中受保护工具后安全拒答"）。

**可复现性**：`corpus.json` 只存来源 URL + 固定版本号 + **SHA-256**；原始文档在 ingest 阶段拉取并校验哈希，写入 `corpus.lock.json` 记录真正入库的字节；dev/test 严格分离，test 只在选定配置后才跑（避免测试集调参污染）。

### 6.2 评测维度

- **检索（文档级宏平均）**：Recall@K、MRR@K、NDCG@K；每个策略先取更宽的 chunk 候选池，再按来源去重后取 K 篇文档打分；同题多次重复运行先在题内平均再聚合。
- **不确定性**：2000 次重采样的 95% 非参数百分位 bootstrap 置信区间；每一策略与 baseline（vector）在**同一批共享题**上做逐题 win/loss/tie 对比 + delta 的置信区间。
- **答案质量**：引用来源对齐率 / 宏平均、证据文档覆盖率、grounded 答案率、拒答正确率、终态结果准确率。
- **Agent 策略**：工具选择正确率（用 `required_all` / `required_any` / `forbidden` 三类标签评估，**允许合法多路径，不做动作序列精确匹配**）、权限拒绝处理正确率、任务成功率。
- **工程指标**：p50/p95 延迟、已知 LLM 成本、执行失败率、工具错误 run 率 / 单次调用失败率、预算收尾率、trace 写失败率。
- **保守性原则（写简历时可强调的严谨性）**：引用「匹配到正确来源文档」只算 *alignment*，**不等于语义蕴含**；语义引用精度、关键事实覆盖、事实一致性三个指标在人工审计表复核前**保持空白**——不允许仪表盘把启发式悄悄变成质量结论。

### 6.3 实测结果 A：检索消融（FastAPI 0.115 冻结测试集，30 题 / 22 题可答；来源：`data/evals/fastapi-0.115/results/*/report.md`）

| 策略 | Recall@6 | MRR@6 | NDCG@6 | p95 延迟 | 单题 LLM 成本 | 失败率 |
|---|---:|---:|---:|---:|---:|---:|
| 纯向量 | 0.909 | 0.833 | 0.849 | 6.46 s | — | 6.7% |
| 向量 + 归一化 FTS | **1.000** | 0.977 | 0.978 | **3.52 s** | — | 0.0% |
| RRF | 1.000 | 0.977 | 0.978 | 4.18 s | — | 0.0% |
| 查询重写 + RRF | 0.955 | 0.932 | 0.925 | 6.85 s | ¥0.000451 | 0.0% |
| 查询重写 + 混合 | 1.000 | **1.000** | **0.988** | 5.29 s | ¥0.000436 | 0.0% |

关键结论（可直接作为简历里的「量化成果」）：
- 混合检索把 NDCG@6 从 0.849 提升到 0.978，**同题 delta +0.129，95% CI [0.021, 0.257]**（不含 0，即统计上可区分），且 p95 延迟反而从 6.46s 降到 3.52s。
- 查询重写 + 混合把 MRR@6 推到 1.000、NDCG@6 到 0.988（delta +0.138，CI [0.031, 0.273]），代价是约 1.5× p95 延迟与 ¥0.000436/题——因此**非 LLM 的「向量+归一化 FTS」被保留为默认**，重写作为可选增强。
- 消融中暴露的失败案例被原样保留（如 `rewrite_rrf` 在 `test-body-01`、`test-deploy-01` 上出现 loss），并在报告中展示。纯向量路保留 2 次瞬时 Supabase RPC 超时，**没有隐藏**。

### 6.4 实测结果 B：多域基准端到端 Agent 评测（tech-multidomain-v1.1，主结果）

**运行配置**（`artifacts/evals/tech-multidomain-v1.1/20260813T080938Z/run.json`）：`kb_only` profile（只暴露 `rag_list_documents / rag_search / rag_read`，按标签排除 8 道受保护抓取用例 → **112 题**）、test 冻结集、1 次重复、`deepseek-v4-flash` + `qwen3.7-text-embedding`、`top_k=6`、关闭 DB 权限覆盖。**112 题全部为真实 Agent 端到端运行**（非检索占位）。

> 口径提醒：全量 120 题 = 112（本次 kb_only 跑）+ 8（受保护抓取，属 `permission` profile，见 §6.5）。同理 32 道拒答题 = 本次 24 道 + 8 道 permission 拒答，所以下面的「拒答正确率 75.0%」分母是 **24**，不是 32。

**自动指标（本次运行的完整结果）**

| 类别 | 指标 | 结果 |
|---|---|---|
| 答案质量 | Grounded 答案率（88 可答题） | **97.7%**（86/88） |
| | 拒答正确率（24 拒答题） | 75.0%（18/24） |
| | 终态结果准确率（可答+拒答合并） | 92.9% |
| | 引用来源对齐精度 | 81.8% |
| | 证据文档覆盖率 | 84.1% |
| Agent 策略 | 工具选择正确率 | **97.0%** |
| | 任务成功率（仅 33 道带期望标注的用例） | 81.8%（27/33） |
| | 拒答终态占比 | 17.9% |
| 工程 | 执行失败率 | **0.9%**（1/112） |
| | 工具错误 run 率 / 单次调用失败率 | 7.1% / 1.1% |
| | 预算收尾率（触发无工具收尾） | 3.6% |
| | Trace 写失败 | 0（0.0%） |
| 成本与延迟 | p50 / p95 延迟 | 49.8 s / 216.1 s |
| | 单题平均 LLM 成本 / 全量总成本 | ¥0.0212 / ¥2.3569（112 题） |

**我从原始 `records.jsonl` 另外统计的实现细节（可用于说明 Agent 行为）**
- 112 题共发生 **801 次工具调用**：`rag_read` 364、`rag_search` 362、`rag_list_documents` 75。即平均每题约 7.2 次工具调用，且 read 与 search 基本 1:1 —— 说明 Agent 实际执行了「先搜再读原文」的取证行为，而不是只看检索预览就作答。
- `answer_status` 分布：`grounded` 91 / `insufficient_evidence` 20 / 执行失败无答案 1。
- 平均单题耗时 67.6 s，最长 293 s（受单题多轮检索 + DeepSeek 调用影响）。

**语义复核结果（重要：由 Codex 完成的 AI 复核，不是真人审核）**

原始报告里 `human_audit` 三个槽位由一次 AI 逐题复核填充（报告已把该槽位显式更名为 `ai_audit`，并在页首注明 "AI review by Codex, not human review"）：

| 指标 | AI 复核结果 | 分母说明 |
|---|---:|---|
| 文档级引用支持率 | **98.0%**（245/250） | 264 条引用中 14 条无法判定，不计为通过 |
| 关键事实覆盖 | **95.8%**（91/95） | 按 95 条基准事实微平均 |
| 全体回答事实一致性 | 94.4%（101/107） | 6 条发现错误，5 条未知 |
| 仅可答题事实一致性 | 92.9%（78/84） | 88 道可答题中 4 条未知、6 条发现错误 |
| 不可答题的语义拒答/澄清 | **95.8%**（23/24） | 1 道执行失败无答案 |

> ⚠️ 引用时**必须标注这是 AI 复核、非人工复核**：没有第二位独立审核者；且因为原始 records 未保存检索 chunk 正文，复核是拿**完整固定版本源文档**比对，比「Agent 当时真正看到的片段」更宽松，**不能表述为 chunk 级精确引用审核通过**。14 条引用与 5 条一致性判定保留为 unknown，null 不计为通过。

**复核暴露的真实问题（值得在简历/面试中作为"我知道它错在哪"的证据）**
- 2 道可答题被错误拒答：`test-deploy-01`（其中基准自身证据也有缺口）、`pyd-test-04`。
- 2 条关键事实漏答：`test-multi-02`、`pyd-test-19`。
- 4 处正文事实错误集中在 SQLAlchemy 域（`sql-test-03/14/16/19`），另有 CORS 过度推断（`test-cors-02`）、K8s 忽略 terminating Pod（`kube-test-10`）。
- **一个重要发现**：多处错误只出现在 `answer` 正文，`claims` 反而是准确的——说明当前 verifier 校验通过**不等于正文无额外错误**，这是已知局限与后续改进方向。

**三次 v1.1 运行的稳定性观察**（同为 `kb_only` / 112 题，可说明 Agent 的非确定性）

| 运行 | Grounded 率 | 拒答正确率 | 引用对齐 | 证据覆盖 | 工具选择 | 执行失败率 | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20260812T075139Z` | 95.5% | 79.2% | 81.8% | 65.9% | 93.9% | 4.5% | 43.9 s / 145.5 s |
| `20260813T024003Z` | 100.0% | 75.0% | 77.3% | 80.7% | 100.0% | 0.0% | 39.9 s / 90.4 s |
| `20260813T080938Z`（最新，含复核） | 97.7% | 75.0% | 81.8% | 84.1% | 97.0% | 0.9% | 49.8 s / 216.1 s |

> 引用建议：**报最新一次（含复核）**；若要强调稳定性，可说明「同配置 3 次重复运行中 grounded 率 95.5%–100%、执行失败率 0%–4.5%」，而不是把单次最优数字当成稳定能力。

### 6.5 实测结果 C：权限链路与 120 题全量覆盖

> ⚠️ 本节修正一个我此前的错误判断：`permission` profile **已经在 v1.1 上跑过**（2026-09-14），并额外产出了一个 120 题合并覆盖套件。

**权限 profile 单跑**（`artifacts/evals/tech-multidomain-v1.1/20260914T064204Z/`）：只暴露 `web_fetch`，确认回调固定返回拒绝，8 题、1 次重复、同模型。

| 指标 | 结果 |
|---|---|
| 工具选择正确 | 8/8（100%） |
| 未授权拒绝处理正确 | 8/8（100%） |
| 拒答正确率 / 终态结果准确率 | 100% |
| 任务成功率 | 8/8（100%） |
| 执行失败 | 0/8 |
| Trace 写失败 | 0（0.0%） |
| p50 / p95 延迟 | 3.61 s / 5.33 s |
| 单题成本 | ¥0.001380 |

**行为证据（从原始 records 统计）**：8 题共发出 8 次 `web_fetch` 调用，**全部**返回 `permission denied: requires confirmation (not granted)`；8 题的 `answer_status` 与 `verification_status` 全部为 `insufficient_evidence`。报告里的 `tool-call failure = 100%` 与 `success=false` 是**预期行为**——任务要求正是「尝试受保护工具 → 被拒 → 诚实拒答」，不是任务失败。

**120 题覆盖套件**（`coverage-test-120-20260914/`）：把 2026-08-13 的 kb_only 112 题与 2026-09-14 的 permission 8 题**合并**，恰好覆盖冻结 test 的全部 **120 个不同 case_id（无缺题、无重复）**，并逐题附带 AI 审核标签（120 条）。两个 profile 的分数**分别报告**，不做成一个总平均分。

| profile | 工具选择 | 权限拒绝处理 | 拒答正确 | 任务成功 | 执行失败 | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| `kb_only`（112 题） | 97.0% | — | 75.0% | 81.8% | 0.9% | 49.8 s / 216.1 s |
| `permission`（8 题） | **100%** | **100%** | **100%** | **100%** | 0.0% | 3.61 s / 5.33 s |

**8 道权限用例的逐题行为**（`test-negative-06/07`、`pyd-test-neg-06/07`、`kube-test-neg-06/07`、`sql-test-neg-06/07`，每道 1 次 `web_fetch`）：全部为「说明抓取未获授权、未观测页面内容、不编造摘要」，并给出「提供正文或授予授权」的下一步；其中 `sql-test-neg-07` 还额外被确认**没有退而改用检索或猜测**来绕过权限。

> **口径说明（引用时必须保留）**：这是**两次不同时间运行的合并覆盖**，不是同一轮的 120 题重跑。合并的依据是两次记录的 benchmark/config 哈希、LLM 模型名、embedding 模型名与 corpus revision 相同——但这**不能证明服务商后端或所有环境变量完全一致**。此外，首次权限运行 `20260914T064118Z` 因沙箱 DNS/网络失败（8/8 `LLMError: deepseek request failed: All connection attempts failed`）已被**显式排除**并保留原始记录供诊断，未混入正式结果。

### 6.6 Agent 评测的可复现控制（`agent_profiles.py`）

用固定工具面 + 用例过滤 + 确认策略，把「知识库质量」「权限拒绝」「受批网络访问」三类信号**彻底隔离**，防止能力泄漏污染分数：

- `kb_only`：只暴露 `rag_list_documents / rag_search / rag_read`，排除 permission 标签用例 —— 主 RAG 分数，web 能力不参与。
- `permission`：只暴露 `web_fetch` 且确认恒为拒绝 —— 考察「选中受保护工具 → 安全拒答」。
- `web_approved`：只暴露 `web_fetch`，确认回调**只放行调用方指定的精确主机白名单** —— 独立验收套件，绝不授予全局网络访问。
- `full`：保留历史 CLI 行为。

技术实现：`AgentRuntime.restrict_tools(allowlist)` 只裁掉暴露给 LLM 的 schema，底层服务不动——模型若"凭空调用"被裁掉的工具会被正常拒绝。

### 6.7 产物与回归

每次运行产出 `records.jsonl`（逐题原始记录）、`summary.json`、`report.md`、`index.html` 仪表盘、`comparison.svg` 对比图，以及（Agent 评测时）待人工填写的 `audit.template.jsonl`；人工复核后用 `pkb-agent eval report` 重新打分。

---

## 七、技术栈清单

| 层 | 选型 |
|---|---|
| 语言 | Python 3.12 |
| LLM | DeepSeek Chat Completions，原生 `tools` 调用，模型 `deepseek-v4-flash`，JSON Output |
| 向量库 | Supabase Postgres + `pgvector`（`<=>` 余弦） |
| 词法检索 | Postgres 全文检索（`tsvector` + `ts_rank` + `websearch_to_tsquery`） |
| 数据库客户端 | `supabase-py`（`rpc()` 调用 SQL 检索函数） |
| Embedding | DashScope 原生 SDK，`qwen3.7-text-embedding`，1536 维，batch=20 |
| Web 搜索 | Tavily / Serper / Bing（环境变量切换） |
| 后端服务 | FastAPI + SSE + uvicorn |
| 前端 | Next.js + TypeScript |
| CLI | Typer + Rich |
| 文档解析 | pypdf / BeautifulSoup + lxml |
| 分词统计 | tiktoken（cl100k_base） |
| 可靠性与测试 | tenacity（重试）、pytest + pytest-asyncio + respx、ruff、mypy |
| 数据库 | 9 个迁移：扩展 / 核心表 / 检索函数 / trace 表 / 记忆表 / 权限表 / RLS 策略 / prompt 溯源 / 可验证答案 |

---

## 八、可直接改写成简历条目的草稿

> 以下为不同长度的版本，按投递岗位篇幅取用。**建议优先保留带数字的两条**。

### 版本 A（3 条 · 标准长度）

- **ReAct 个人知识库 Agent（Python / DeepSeek / Supabase）** —— 独立设计并实现基于 Supabase + pgvector 的混合检索知识库 Agent，用 DeepSeek 原生 Function Calling 驱动 Thought→Action→Observation 循环；运行时与外部世界解耦，**只能经带权限与全链路追踪的工具层访问数据与网络**，约 10.8k 行 Python，25 个测试文件。
- **设计并落地「可验证回答」机制**：最终输出被约束为 `answer + claims(fact/inference) + citations` 的机器校验 JSON 契约，引用 id 必须存在于本轮检索构建的内存证据台账，伪造/过期/孤立引用直接拒绝并触发有界修复，修复失败则返回确定性拒答——从机制上消除幻觉引用。
- **构建评测驱动的 RAG 优化闭环**：搭建 41 篇固定版本语料 / 120 题冻结测试集（含多文档、同义改写、术语歧义、版本陷阱、拒答、受保护工具六类挑战）与 5 策略检索消融；**混合检索将 NDCG@6 从 0.849 提升至 0.978（同题 Δ+0.129，95% CI [0.021, 0.257]），p95 延迟由 6.46s 降至 3.52s**，并给出 2000 次 bootstrap 置信区间与逐题 win/loss/tie。
- **完成 120 题全量 Agent 评测（双 profile 覆盖）**：知识库 112 题上 Grounded 答案率 **97.7%**、工具选择正确率 **97.0%**、执行失败率 **0.9%**、Trace 写失败 **0**；受保护工具链路 8 题上「工具选择 / 未授权拒绝处理 / 拒答正确 / 任务成功」全部 **8/8（100%）**、0 执行失败；112 题共 801 次工具调用（search/read 约 1:1）。语义复核显示引用支持率 98.0%、关键事实覆盖 95.8%，并逐题定位了 2 道错误拒答与 6 处正文事实错误（含"claims 准确但 answer 有误"的 verifier 盲区）。

### 版本 B（6 条 · 技术深度型，适合 Agent/RAG 岗）

1. 基于 **ReAct** 范式实现 Agent 运行时：DeepSeek 原生工具调用，步数预算 + 预留无工具收尾预算，预算耗尽退化为「证据不足」拒答而非报错；单轮多工具调用、提供方协议补齐、异常兜底均已处理。
2. 实现 **pgvector 语义检索 + Postgres 全文检索**双路召回（候选池 3×K、DB 端 SQL 函数 RPC），并实现 **RRF** 与**归一化加权分数融合**两种混合策略；针对自然语言提问做了宽容 OR 词法投影，解决 `websearch_to_tsquery` 相邻词 AND 导致零召回的问题。
3. 设计**三层工具权限体系**（allow/ask/deny + 参数级约束 + `tool_permissions` 表动态覆盖 + 读取失败回退 YAML），配套 **SSRF 防护**（私网/环回/链路本地/云元数据端点拦截、scheme 白名单）；`ask` 类工具在非交互模式默认拒绝，Web 侧通过审批代理实现 human-in-the-loop。
4. 实现**查询重写**（前置 prompt 阶段、温度 0、JSON 输出、多 query 并发检索取最优、失败回退原 query、按 (tool, query, question) 缓存且缓存命中不重复计费）与**prompt 生命周期清单**（按能力声明而非文件名推断）。
5. 实现**非致命可观测性**：trace 参数/结果/错误全量脱敏，仅对瞬时传输故障做有界指数退避重试，写失败降级为「可观测性降级」计数而不丢弃已完成答案；成本按已完成 usage 分 cache-hit/miss 计价。
6. **跑通 v1.1 冻结集全量 Agent 评测（双 profile，共 120 题）并做逐题语义复核**：知识库 profile 112 题 Grounded 答案率 97.7%、拒答正确率 75.0%、工具选择正确率 97.0%、执行失败率 0.9%、p50/p95 49.8s/216.1s、¥0.0212/题、801 次工具调用（rag_read 364 / rag_search 362 / rag_list_documents 75）；受保护工具 profile 8 题「工具选择 / 未授权拒绝处理 / 拒答正确 / 任务成功」全 100%、0 执行失败、p95 5.33s。逐题复核定位了错误拒答与正文事实错误，并发现「claims 准确但 answer 有误」的 verifier 校验盲区。

### 版本 C（一句话版）

独立开发评测驱动的 ReAct 个人知识库 Agent：Supabase+pgvector 混合检索（NDCG@6 0.849→0.978）、DeepSeek 原生工具调用、可验证回答 JSON 契约（引用伪造即拒答）、动态工具权限与 SSRF 防护；配套 120 题冻结测试集、5 策略消融 + bootstrap 置信区间，并完成 120 题全量端到端 Agent 评测（知识库 112 题 Grounded 97.7% / 工具选择 97.0% / 执行失败 0.9%，受保护工具 8 题全项 100%）。

---

## 九、面试可能被追问的点（提前备好答案）

1. **为什么引用只让模型给 id，不让它给标题/URL？** 因为「选择证据」和「编造证据元数据」是两件事。元数据由运行时从真实工具观测生成，模型只能选，不能造——这是把引用可信度从「提示词约束」提升到「代码不可绕过」。
2. **为什么 `web_search` 摘要不能当引用？** 摘要没有来源语境、无法验证内容哈希与时效。必须 `web_fetch` 取回原文，才记录抓取时间/域名/信任等级/内容 SHA-256/截断状态。区分「听说」与「读到」。
3. **RRF 和加权分数融合怎么选？** RRF 只用排名、天然免归一化，适合两路分数量纲不可比的场景；加权融合保留分数量级信息。实测在本语料上两者 NDCG 持平（0.978），但加权混合延迟更低（3.52s vs 4.18s），故生产默认用加权混合，RRF 留作对照。
4. **为什么测试集要冻结？** 检索策略有大量可调超参（top_k、rrf_k、权重、chunk 大小）。只在 dev 上调参，test 只在选配置后跑一次，否则指标是过拟合出来的。
5. **bootstrap 置信区间解决什么？** 22 个可答案例的均值差异可能纯属偶然。2000 次重采样给出区间，CI 不含 0 才敢说「有区分度」——避免在小样本上宣称提升。
6. **为什么人工指标留空？** 「引用到了正确文档」是 source match，不是语义蕴含（entailment）。不做人工复核就报「语义引用精度」等于把启发式包装成结论。宁可留空。
7. **工具预算耗尽了怎么办？** 不报 max-step 错误，而是预留 3 步、注入收尾指令、在无工具前提下走正常的校验+修复流程，最终仍可能得到 grounded 答案或确定性拒答。
8. **权限改动怎么保证运行时生效？** 默认每次工具调用前刷新 `tool_permissions` 覆盖表；一行整条替换该工具规则（不做局部合并，行为可预测）；读取失败清空动态规则回退 YAML，绝不留下过期规则生效。
9. **Trace 写失败会不会丢答案？** 不会。幂等写入 + 仅瞬时错误重试 + 写失败仅累加降级计数，答案路径继续。评测把它单列为可观测性降级。
10. **怎么防止能力泄漏污染评测？** Agent 评测用固定 profile 裁工具面 + 用例标签过滤 + 固定确认策略，`restrict_tools` 只裁 LLM 可见 schema，凭空调用被裁工具会被正常拒绝。
11. **怎么证明权限拒绝链路真的有效？** 专门用 `permission` profile 单跑 8 道用例：只给 `web_fetch`、确认回调固定拒绝。结果是 8 次 `web_fetch` 全部返回 `requires confirmation (not granted)`、8 题全部落到 `insufficient_evidence` 拒答，且逐题确认模型没有「退而改用检索或猜测」来绕过权限——工具选择/拒绝处理/拒答/任务成功 4 项 8/8。注意报告里 `tool-call failure=100%` 是**预期**的拒绝，不是失败。
12. **为什么 120 题要分两个 profile 报？** 因为把「知识库答得好」和「受保护工具被正确拒绝」混成一个平均分会掩盖两类完全不同的能力。而且这份 120 题覆盖是两次不同时间运行的合并，不是同轮重跑——合并只依据记录到的哈希/模型/语料版本一致，不能证明服务商后端与环境变量完全一致，所以必须声明清楚。

---

## 十、可提及的局限（诚实加分项，不要说满）

- **5 策略检索消融只在 FastAPI 切片（10 篇文档、30 题）上跑过完整对比**；多域基准 v1.1 的三次运行都是 Agent 端到端评测（`strategies=[]`），**没有在 v1.1 语料上重跑全量检索消融**。结论不宣称可泛化到任意语料。
- v1.1 的 120 题覆盖是**两次不同时间运行的合并**（112 题 kb_only @2026-08-13 + 8 题 permission @2026-09-14），**不是同一轮 120 题同时重跑**；哈希/模型名/corpus revision 相同只说明记录到的配置一致，不证明服务商后端与环境变量完全一致。
- **`web_approved`（批准后真正放行抓取）这条链路有意未覆盖**（它需要真实外网访问，当前沙箱环境不具备）。因此权限测试只覆盖「未确认时拒绝抓取」，**不代表已测通过允许访问、其他授权模式或全部安全边界**——简历/面试中不要暗示已覆盖放行路径。
- 首次权限运行 `20260914T064118Z` 因沙箱 DNS/网络失败（8/8 DeepSeek 连接失败）被显式排除；保留原始记录供诊断，未计入正式结果。
- **语义指标来自一次 AI 复核（Codex），不是人工复核，也没有第二位独立审核者**；且因原始 records 未保存检索 chunk 正文，复核以完整固定版本源文档为比对基准，比 Agent 实际看到的片段更宽松——**不能表述为 chunk 级精确引用审核通过**。14 条引用判定与 5 条一致性判定为 unknown，不计为通过。
- 语义对齐 ≠ 语义蕴含：报告里的 `citation_alignment_precision` 只表示「引用落到了标注的相关文档」，不是「这句话被文档证明」。
- 复核发现多处事实错误**只出现在 `answer` 正文而 `claims` 准确**，说明当前 verifier 通过不等于正文无额外错误——这是已知的校验盲区。
- 三次同配置运行的 grounded 率在 95.5%–100%、执行失败率 0%–4.5% 之间波动，**单次数字不代表稳定能力**；p95 延迟（90s–216s）波动同样较大。
- 成本是「已知 LLM usage」的计算，不含 embedding 提供方账单，**不是完整发票**。
- 前端工作台只在本地双进程（uvicorn:8000 + next:3000）开发模式下运行，未做生产部署与鉴权加固；数据隔离依赖 Supabase RLS 策略 + user_id 维度，尚未接入真正的多租户认证。
