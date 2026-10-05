# Real CLI demo / 真实 CLI 演示

[Download and watch / 下载观看视频](https://raw.githubusercontent.com/FfszHy/ReAct_KB_Agent/main/docs/media/demo.mp4)
· [Chinese captions](demo.zh-CN.srt)
· [Provenance and verification](provenance.json)

The link downloads the MP4 (9.2 MiB) for local playback.

Recorded on 2026-10-05 (Asia/Shanghai). About 4 min 36 sec, 2560×1600 H.264,
no audio. Captions describe visible actions and outcomes; they are editorial
annotations, not an audio transcript. The original 802 frames and their
presentation timestamps are retained at normal speed. Only the external black
margin and window title bar are cropped. Captions and chapter labels sit
outside the terminal content. The recording begins during the first catalog
query and includes the final results of all five scenarios.

| Time | Observed scenario |
|---|---|
| 00:00–00:46 | Catalog pagination; 41 documents counted, with topic grouping labelled as model inference. |
| 00:46–01:44 | FastAPI and Pydantic retrieval, source reading, answer and citations. |
| 01:44–02:20 | A request for measured production p99; insufficient measurement evidence, followed by an explicit evidence-boundary response. |
| 02:20–03:20 | User enters `y` to authorize fetching. The first review rejects an unsupported `HTTP 200` assertion; the revised answer passes the second review. |
| 03:20–04:36 | Autonomous ConfigMap retrieval. The first review rejects an overly exclusive statement about volume updates; the revised answer and citations are shown. |

The fourth case was **approved**, although the recording script suggests `n`
when demonstrating refusal. The two repair sequences occurred in this recording;
they were not reconstructed from historical preflight runs.

The examples show retrieval, evidence boundaries, tool authorization, and two
visible review-and-repair sequences. A passed AI review remains fallible. These
examples do not establish general reliability, and the ConfigMap case did not
inspect or repair an actual Kubernetes cluster.

`demo.mp4` is the smaller publication export (H.264 CRF 26); the full-quality
export and original `.mov` remain local. It has passed full-file decoding, and
the source frame count and timestamps were compared before publication.
