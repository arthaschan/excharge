#!/usr/bin/env bash
# 批量下载论文全部 17 篇参考文献 PDF 到 references/ 目录（Linux 版）
# arXiv 开放获取优先；Nature Comm / EU AI Act 用官方 OA 链接。
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p references
cd references || exit 1

dl() {
  local name="$1"; local url="$2"
  local f="${name}.pdf"
  if [ -f "$f" ]; then
    local sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt 20000 ]; then echo "[SKIP] $f ($((sz/1024)) KB) 已存在"; return; fi
  fi
  echo "[DL] $name <- $url"
  curl -sL --max-time 120 -A "Mozilla/5.0" -o "$f" "$url"
  local sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -lt 20000 ] || ! head -c 5 "$f" | grep -q "%PDF"; then
    echo "  [FAIL] $name ($sz bytes, 非有效PDF)，删除"
    rm -f "$f"
  else
    echo "  [OK] $f ($((sz/1024)) KB)"
  fi
}

dl "[01]_USAD"                     "https://arxiv.org/pdf/2001.04384"
dl "[02]_GDN"                      "https://arxiv.org/pdf/2101.02310"
dl "[03]_Yang_NatCommun"           "https://www.nature.com/articles/s41467-025-67703-7.pdf"
dl "[04]_IsolationForest"          "https://arxiv.org/pdf/1811.02141"
dl "[05]_AnomalyTransformer"       "https://arxiv.org/pdf/2110.02642"
dl "[06]_OmniAnomaly"              "https://arxiv.org/pdf/1905.08730"
dl "[07]_AttentionNotExplanation"  "https://arxiv.org/pdf/1902.10186"
dl "[08]_TreeSHAP"                 "https://arxiv.org/pdf/1905.04610"
dl "[09]_SHAP"                     "https://arxiv.org/pdf/1705.07874"
dl "[10]_ModelCards"               "https://arxiv.org/pdf/1810.03993"
dl "[11]_EU_AIAct"                 "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"
dl "[12]_GradCAM"                  "https://arxiv.org/pdf/1610.02391"
dl "[13]_IntegratedGradients"      "https://arxiv.org/pdf/1703.01365"
dl "[14]_MTAD_GAT"                 "https://arxiv.org/pdf/2009.02040"
dl "[15]_IsmailFawazReview"        "https://arxiv.org/pdf/1809.04356"
dl "[16]_TimeVQVAE_AD"             "https://arxiv.org/pdf/2311.12550"
dl "[17]_TabPFN2.5"                "https://arxiv.org/pdf/2511.08667"

echo ""
echo "=== 下载结果 ==="
ls -la *.pdf 2>/dev/null
echo "共 $(ls *.pdf 2>/dev/null | wc -l) 个 PDF"
