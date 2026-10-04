# 自主 ReAct 候选演示：预检审阅

审阅日期：2026-10-04（Asia/Shanghai）。结论：可以展示真实的工具选择和跨轮观察依赖；两次预检的最终回答均有语义问题，暂不能作为“完整正确地完成任务”的成功示范。没有继续反复试跑挑选成功样本。

本审阅由 Codex 对原始记录进行额外 AI 语义复核，不能标注为人工审核。

## 候选问题与录制范围

问题见 [react-configmap.txt](demo-prompts/react-configmap.txt)。它提供假设场景和目标，不指定工具、文档、检索顺序或期望结论：评估“修改 ConfigMap 后等一分钟就会全部自动更新”是否可靠，并指出还缺的现场信息。

这是依据固定知识库做判断的任务。本 Agent 没有在这两次运行中访问 Kubernetes 集群、查看实际 Pod 配置或执行修复命令；不能称为已定位或修复真实故障。

两次运行均为真实 CLI 预检，保存了运行状态、工具动作和观察；**都不是 Ghostty 窗口录屏**。之后在 Ghostty 执行仍会实时调用模型，动作和回答可能变化，不能保证复现这些步骤。

| 记录 | 北京时间 | Run ID | 工具调用 |
|---|---|---|---|
| [首次预检](../artifacts/ghostty-recording/react-configmap-preflight-initial.json) | 13:19:37–13:20:00 | `5a410ba7-5821-4763-8c61-0d4ae07b98ca` | 搜索 3 次、读取 4 次 |
| [缩小任务后的预检](../artifacts/ghostty-recording/react-configmap-preflight.json) | 13:21:24–13:21:43 | `605d5507-4ba3-4cdf-8f0d-78f27512abfe` | 搜索 1 次、目录 1 次、读取 5 次 |

首次问题还要求“最小处理方案”；第二次去掉修复方案要求，聚焦判断与缺失信息。两次问题不同，不能当作同题重复评测或成功率统计。原始记录保留了有问题的回答，没有改写成预期答案。`artifacts/` 为本地证据目录，默认不随仓库发布；公开本报告时应同时附上经过隐私检查的记录，否则读者无法访问上表的本地证据链接。

## 第二次运行：能核验的反馈链

以下轮次来自记录中的 `model_turns`，只计算主 Agent 的模型调用，不把检索前的查询改写计为额外 ReAct 轮次。`step` 使用原始记录的从 0 开始编号。

| 主 Agent 模型轮次 | 本轮选择的动作 | 观察与下一轮的关系 |
|---|---|---|
| 1 | step 0 `rag_search`；step 1 `rag_list_documents` | 搜索结果给出 ConfigMaps 的文档 ID 和相关 chunk ID。目录只返回 SQLAlchemy 的前 10 个条目，没有证据表明这次目录调用推进了当前任务。两个动作由同一轮模型提出。 |
| 2 | step 2 按文档 ID `rag_read`；step 3 读取 Secrets chunk | 两个读取 ID 均已出现在轮 1 的搜索观察中。文档读取返回 5 个 chunk 的标识，但正文因观察长度限制被截断。 |
| 3 | step 4、5、6 分别读取 ConfigMaps 的三个 chunk | step 4 使用的新 chunk ID 首次来自轮 2 文档读取；step 5、6 的 ID 在轮 1 和轮 2 均已出现。这三个动作同属一个批次。 |
| 4 | 无工具调用，提交最终答案 | 输出结构校验通过，但额外 AI 语义复核发现下面列出的缺陷。 |

两段明确的跨轮依赖：

1. 轮 1 / step 0 的搜索观察含文档 ID `ac61c059-5efd-43e9-8c3c-be3c811c9a7b`；轮 2 / step 2 使用这个 ID 读取 ConfigMaps 文档。目录结果没有这个 ID。
2. 轮 2 / step 2 的 `citation_evidence` 首次给出 chunk ID `61a7b5cb-aa3f-4a03-87da-a29ceb1510b5`；轮 3 / step 4 使用它读取 immutable ConfigMap 段落。该 ID 不在轮 1 的任何工具观察中。

这支持“Agent 根据工具返回的具体标识继续选择读取动作”。文档观察被截断后又按 chunk 读取，也与继续补取局部原文相一致；记录没有保留内部推理，不能断言其动机一定是主动识别截断。step 4 的 immutable 内容没有进入最终论证，也不能把每个调用都说成必要或高效。

准确计数是 **3 轮工具选择、7 次工具调用、1 轮最终回答**。同批工具虽然顺序执行，下一次模型决策要等本批返回后才发生；不能说成“7 轮自主思考”。首次预检的 `model_turns` 为空是采集方式的缺陷，不表示模型没有多轮调用，也不能据此反推精确轮次。

## 语义审阅

这两次历史记录均返回 `status=finished`、`answer.status=grounded`、`verification.status=verified`，且没有 JSON 修复。这些状态确认执行与输出结构校验通过，**不等于每句话均被原文支持**。当时版本的校验器检查 JSON、声明和本轮证据 ID 等条件，没有对自由答案正文逐句做语义蕴含检验。之后的通用修复和复跑见 [Agent 质量修复记录](agent-quality-fixes.md)，不改变本报告中历史运行的评价。

### 有原文支持的判断

两次运行均读取到 ConfigMaps chunk `65c882fc-a8f2-4db2-b730-e6a77bea5e73`，其中明确说明：普通卷投影最终更新；总延迟受 kubelet 同步周期与缓存传播时延影响；环境变量不自动更新且需要重启 Pod；`subPath` 挂载不接收 ConfigMap 更新。因此“一分钟内全部生效”不能作为普遍保证，这一核心判断有依据。

没有现场消费方式、同步设置与应用读取行为的信息，不能据此确认假设中的根因。卷内文件更新和应用已经使用新值也应分开判断。

### 首次预检的问题

- 最终答案把环境变量、`subPath`、immutable 合并为“重建相关 Pod 是唯一有效手段”，超出了原文。immutable 原文要求删除并重建 ConfigMap；仅重建 Pod 并不足以改变原有配置内容，也没有依据断言唯一方案。
- 声称“一分钟自动更新”只在卷挂载场景可能成立，遗漏其已读取文档中的应用直接调用 Kubernetes API、订阅变化的方式。
- 对“Deployment 正常”的 Pod 模板与副本状态解释没有取得 Deployment 文档证据。
- 按 Unicode 字符计，正文 660 字符；提示要求控制在 400 字以内。实际录屏阅读负担比要求更大。

### 第二次预检的问题

- 正文开头写“只有以卷挂载的 ConfigMap 会最终自动更新”，把卷投影的行为扩展成排他性结论。固定文档还描述了应用通过 API 订阅更新；首次预检的完整 chunk `15fa3aaf-10cb-4d6e-8240-a7f92acf9de9` 可核验这一点。第二次文档读取在 API 订阅句中途被截断，更不能从有限观察推出“只有”。
- 正文写“修改 ConfigMap 不改动 Deployment 的 Pod 模板，故 Deployment 显示正常并不代表 Pod 已重建”，本轮没有取得 Deployment 文档证据；这句话也没有出现在带引用的 `claims` 中。即使某项知识通常成立，也不能包装成已由本轮引用核验的内容。
- 正文强调消费方式和 kubelet 设置，但没有说明应用是否重新读取配置也是现场判断的重要限制。
- 按 Unicode 字符计，正文 387 字符；提示要求控制在 250 字以内。

## 录屏与讲解建议

这一段可作为“自主选择动作及其质量边界”的候选演示，不应预先宣布最终答案正确。录制时完整保留真实输出；如果再次出现过度概括，直接指出，不能删掉错误句后展示为完整成功。

可用的讲解是：“我只给出目标，没有指定检索步骤。这里它从搜索结果取到了文档 ID，读取文档后又使用新返回的 chunk ID 继续读取，形成了跨轮反馈。当前引用结构检查通过，但对答案语义仍需要进一步验证。”

不能宣称：完整内部思考已展示、所有动作均有必要、已经检查或修复真实集群、七次调用就是七轮反馈、或者一个案例证明总体可靠性。查询改写属于检索前处理，不能把一次检索里的多个改写词当作多轮自主决策。

相关实现位置：`src/pkb_agent/agent/runtime.py` 的 `_loop`、`_execute_tool_call`；`src/pkb_agent/prompts/query_rewriter.py` 的 `rewrite`；`src/pkb_agent/agent/verification.py` 的 `AnswerVerifier`；`src/pkb_agent/app/cli.py` 的 `_render_event`。CLI 展示动作和截断观察，不展示完整内部推理。

## 原始记录完整性

审阅时 SHA-256：

```text
eea955ac44f9c201cadf816fe64456b8996d91b6c6fd3702740c4a2615c0b080  react-configmap-preflight-initial.json
921b4780ec7cdc8d41a80180d05eeb92da3c8ba47210b840a709163773817a11  react-configmap-preflight.json
```
