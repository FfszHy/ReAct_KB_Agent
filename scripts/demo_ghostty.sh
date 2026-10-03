#!/usr/bin/env bash
# Run real CLI requests in the user's terminal, with manual pauses for recording.
# This script does not start a recorder, control Ghostty, or replay saved output.
set -euo pipefail

PKB_DEMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKB_DEMO_ROOT"

PKB_DEMO_SCOPE='eval-tech-multidomain-v1.1'
PKB_DEMO_QUESTIONS=(
  '请只用 rag_list_documents 完整翻页查看本用户的知识库目录，按技术领域统计资料数，并报告总数。只需四行分类统计和一行总数，不列全部标题，不检索正文，不推断文档版本。'
  '只依据知识库中的 FastAPI 0.115.0 与 Pydantic v2.10.6 文档，核对两个问题：① 一个 FastAPI 路径同时声明了返回类型注解和 response_model，实际响应处理以哪个为准？② Pydantic v2 模型中写 nickname: Optional[str]，但不写默认值，创建模型时能否省略 nickname？请先检索并读取相关段落，两点分别给出处，回答控制在 250 字以内；不要联网，不扩展到其他主题。'
  '仅根据本知识库，能否给出我部署的 FastAPI 用户创建接口在每秒一万次请求下的实测 p99 延迟，并提供真实压测记录作为出处？不要估算或联网。如果没有足够测量证据，请在两句话内说明无法确定，不必叙述检索过程。'
  '请用 web_fetch 抓取 https://raw.githubusercontent.com/fastapi/fastapi/0.115.0/docs/en/docs/tutorial/background-tasks.md ，仅依据这次成功抓取的页面，说明 BackgroundTasks 在响应前还是响应后运行任务。如果没有获准抓取或抓取失败，请说明本次无法核验，不要用模型记忆或知识库代替这次网页内容。'
)
PKB_DEMO_TITLES=('知识库目录' '跨文档问答与引用' '没有实测数据时的回答' '网页抓取权限')

show_command() {
  local index="$1"
  if [[ "$index" == 3 ]]; then
    printf 'PERMISSIONS_DB_OVERRIDES_ENABLED=false '
  fi
  printf "conda run --no-capture-output -n pkb-agent pkb-agent ask '%s' --user %s\n" \
    "${PKB_DEMO_QUESTIONS[$index]}" "$PKB_DEMO_SCOPE"
}

case "${1:-}" in
  --list)
    for PKB_DEMO_INDEX in 0 1 2 3; do
      printf '\n%s / 4 · %s\n' "$((PKB_DEMO_INDEX + 1))" "${PKB_DEMO_TITLES[$PKB_DEMO_INDEX]}"
      show_command "$PKB_DEMO_INDEX"
    done
    exit 0
    ;;
  --check)
    command -v conda >/dev/null || { printf '未找到 conda。\n' >&2; exit 1; }
    conda run -n pkb-agent pkb-agent eval validate data/evals/tech-multidomain-v1.1
    exit 0
    ;;
  ''|1|2|3|4) ;;
  *) printf '用法：bash scripts/demo_ghostty.sh [1|2|3|4|--list|--check]\n' >&2; exit 2 ;;
esac

if [[ ! -t 0 || ! -t 1 ]]; then
  printf '请在 Ghostty 中直接运行，让键盘输入和输出保留在终端。\n' >&2
  exit 1
fi
command -v conda >/dev/null || { printf '未找到 conda。\n' >&2; exit 1; }

printf '\nPKB-Agent · Ghostty 实录\n'
printf '下面会实时调用模型和数据库。每段由你按回车开始。\n'
printf '请先开始屏幕录制；Ctrl+C 随时结束。\n'
printf '若出现工具确认，请自行判断；第 4 段请手动输入 n 展示拒绝路径。\n'

PKB_DEMO_FIRST=0
PKB_DEMO_LAST=3
if [[ -n "${1:-}" ]]; then
  PKB_DEMO_FIRST=$(($1 - 1))
  PKB_DEMO_LAST="$PKB_DEMO_FIRST"
fi
for ((PKB_DEMO_INDEX=PKB_DEMO_FIRST; PKB_DEMO_INDEX<=PKB_DEMO_LAST; PKB_DEMO_INDEX++)); do
  printf '\n%s / 4 · %s\n\n' "$((PKB_DEMO_INDEX + 1))" "${PKB_DEMO_TITLES[$PKB_DEMO_INDEX]}"
  show_command "$PKB_DEMO_INDEX"
  printf '\n按回车执行此命令：'
  read -r PKB_DEMO_INPUT
  if [[ "$PKB_DEMO_INDEX" == 3 ]]; then
    PERMISSIONS_DB_OVERRIDES_ENABLED=false conda run --no-capture-output -n pkb-agent \
      pkb-agent ask "${PKB_DEMO_QUESTIONS[$PKB_DEMO_INDEX]}" --user "$PKB_DEMO_SCOPE"
  else
    conda run --no-capture-output -n pkb-agent \
      pkb-agent ask "${PKB_DEMO_QUESTIONS[$PKB_DEMO_INDEX]}" --user "$PKB_DEMO_SCOPE"
  fi
  printf '\n本段命令已返回。请检查答案，可向上滚动展示原文事实、推断和引用。\n'
  printf '展示完后按回车继续：'
  read -r PKB_DEMO_INPUT
done
printf '\n演示结束，可以停止屏幕录制。\n'
