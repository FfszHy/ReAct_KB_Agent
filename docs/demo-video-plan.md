# PKB-Agent 演示视频与实际命令

当前录制方式已改为 **Ghostty 窗口直接录屏**，见 [Ghostty 实录说明](ghostty-recording.md) 和 `scripts/demo_ghostty.sh`。下方记录的是之前的排版回放版本，用户未采用；新的实录需要用户启动系统录屏后取得真实画面。

已制作 5 分 09 秒、1080p 横屏、中文字幕、无配音的演示片，适合面试时自行讲解。素材来自真实 CLI 调用；成片是终端记录回放与原始答案摘录，并非连续桌面录屏。

交付目录：`artifacts/demo-video/2026-10-04/`。

- [成片](../artifacts/demo-video/2026-10-04/ReAct_KB_Agent_Demo_1080p.mp4)
- [观看与讲解说明](../artifacts/demo-video/2026-10-04/README.md)
- [全部实际演示命令](../artifacts/demo-video/2026-10-04/demo_commands.sh)
- [试录与证据审阅](../artifacts/demo-video/2026-10-04/EVIDENCE_REVIEW.md)
- [分镜源文件](../artifacts/demo-video/2026-10-04/video_plan.json)

## 主片内容

| 时间 | 展示内容 |
|---|---|
| 00:00–00:26 | 41 份固定官方文档、4 个领域、当前 376 个片段 |
| 00:26–01:08 | Agent 完整翻页统计目录 |
| 01:08–02:45 | FastAPI 与 Pydantic 接口契约问答，展示工具、事实、推断与引用 |
| 02:45–03:30 | 缺少生产压测数据，拒绝编造 p99 |
| 03:30–04:07 | 预设输入 n 拒绝 web_fetch，说明无法核验页面 |
| 04:07–04:25 | 保留输出校验失败与语义误读的已知限制 |
| 04:25–05:09 | 本次工程检查、历史评测口径与收尾 |

## 录制采用的核心问题

在仓库根目录执行，使用 Conda 环境 `pkb-agent`：

```bash
conda run -n pkb-agent pkb-agent doctor
conda run -n pkb-agent pkb-agent eval validate data/evals/tech-multidomain-v1.1
conda run --no-capture-output -n pkb-agent pkb-agent ask \
  "只依据知识库中的 FastAPI 0.115.0 与 Pydantic v2.10.6 文档，核对两个问题：① 一个 FastAPI 路径同时声明了返回类型注解和 response_model，实际响应处理以哪个为准？② Pydantic v2 模型中写 nickname: Optional[str]，但不写默认值，创建模型时能否省略 nickname？请先检索并读取相关段落，两点分别给出处，回答控制在 250 字以内；不要联网，不扩展到其他主题。" \
  --user eval-tech-multidomain-v1.1
```

实际录制中，该问题调用了检索和阅读工具，关键结论及两条引用由本轮观测支持。也保留局限：回答没有严格控制在 250 字以内，FastAPI 引用面板缺少标题而显示真实 UUID。没有后期伪造更漂亮的原始输出。

完整的目录、拒答、权限问题及软件测试命令见 `demo_commands.sh`。权限场景采用 `PERMISSIONS_DB_OVERRIDES_ENABLED=false`，确认出现后输入 `n`；不要使用自动批准选项。真实录制由脚本输入 n，输入时间写在回执里。

## 不能夸大的部分

原先拟展示的复杂字段序列化问题在第一次试录中未完成：最终 JSON 校验失败。聚焦版本虽然通过引用结构校验，仍误解了 Pydantic 的字段排除规则。这些记录均保留，成片没有将它们包装为成功案例。额外语义审阅由 AI 完成，不是人工审核，也不等于运行时自动发现了语义错误。

普通 `ask` 中的“不要联网”是提示要求，不是程序级工具隔离。`Verified citations` 验证本轮引用资格和结构，不能保证所有语义结论正确。257 项软件测试通过，不代表 257 道真实模型问答都正确。

历史评测为 2026-08-13 的 112 题知识库运行与 2026-09-14 的 8 题权限拒绝运行的覆盖汇总。AI 引用支持 245/250（98.0%，另 14 条未知），事实一致性 101/107（94.4%，另 5 题未知）；知识库应拒答题正确拒答 18/24（75%）。未知项排除，分母与来源日期均在视频展示。这些不是录制当天重新跑出的分数。

视频与项目源码尚未由本任务发布到 GitHub。后续可将视频作为仓库 README 或 Release 的附件；不要把全部中间渲染帧纳入源码。项目的开源许可与正式发布资料仍须按实际仓库状态整理，不能由本片替代。
