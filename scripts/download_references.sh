#!/usr/bin/env bash
# 批量下载参考文献 PDF 到 references/ 目录
# 优先从 arXiv 下载（开放获取），命名规范：{简称}_{arxiv编号}.pdf
set -u
cd /Users/arthas/git/excharge/references || exit 1

dl() {
  local name="$1"; local arxiv="$2"
  local url="https://arxiv.org/pdf/${arxiv}"
  if [ -f "${name}_${arxiv}.pdf" ]; then
    echo "[SKIP] ${name}_${arxiv}.pdf 已存在"
    return
  fi
  echo "[DL] ${name} (${arxiv})"
  curl -sL --max-time 90 -o "${name}_${arxiv}.pdf" "${url}"
  local sz=$(stat -f%z "${name}_${arxiv}.pdf" 2>/dev/null || echo 0)
  if [ "$sz" -lt 10000 ]; then
    echo "[FAIL] ${name} 下载失败(${sz} bytes)，删除重试 v 版本"
    rm -f "${name}_${arxiv}.pdf"
    curl -sL --max-time 90 -o "${name}_${arxiv}.pdf" "https://arxiv.org/pdf/${arxiv}v"
    sz=$(stat -f%z "${name}_${arxiv}.pdf" 2>/dev/null || echo 0)
    [ "$sz" -lt 10000 ] && { echo "[FAIL2] ${name} 仍失败"; rm -f "${name}_${arxiv}.pdf"; return; }
  fi
  echo "[OK] ${name}_${arxiv}.pdf ($((sz/1024)) KB)"
}

dl "USAD"                    "2001.04384"
dl "GDN"                     "2101.02310"
dl "AnomalyTransformer"      "2110.02642"
dl "OmniAnomaly"             "1905.08730"
dl "AttentionNotExplanation" "1902.10186"
dl "SHAP"                    "1705.07874"
dl "GradCAM"                 "1610.02391"
dl "IntegratedGradients"     "1703.01365"
dl "MTAD_GAT"                "2009.02040"
dl "IsmailFawazReview"       "1809.04356"
dl "TimeVQVAE_AD"            "2311.12550"

echo "=== 完成 ==="
ls -la *.pdf
