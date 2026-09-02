#!/usr/bin/env python3
"""md → PDF 学术排版构建脚本（中文论文, 支持 GFM 表格/图片/上标/公式）。

流程: markdown → HTML(+CSS+MathJax CDN 公式兜底) → PDF。
渲染后端: weasyprint(主, 纯 Python 无 JS); 失败自动回落 Chrome headless。
要求: python-markdown + weasyprint; Chrome 仅作为兜底(支持 MathJax JS)。
Homebrew 库不在默认 DYLD 路径时, 需带环境变量运行:
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python build_pdf.py [md] [html] [pdf]
默认: paper/paper_draft.md → paper/paper_draft.html → paper/paper_draft.pdf
公式: 正文若含 $...$/$$...$$ 自动注入 MathJax; 无公式则跳过(weasyprint 无 JS,
      渲染 LaTeX 公式需走 Chrome 兜底)。
"""
import re, sys, os, subprocess, tempfile

MD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'paper_draft.md')
BASE = os.path.dirname(MD)
HTML = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, os.path.splitext(os.path.basename(MD))[0] + '.html')
PDF = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, os.path.splitext(os.path.basename(MD))[0] + '.pdf')

import markdown

with open(MD, encoding='utf-8') as f:
    text = f.read()

has_math = bool(re.search(r'\$[^$\n]+\$|\$\$', text))  # 粗略检测行内/块公式

body = markdown.markdown(
    text,
    extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list'],
)

# 图与图注居中: 把"仅含 <img> 的段"与紧随其后的"**图N:** 开头段"包成居中块
body = re.sub(
    r'(<p><img[^>]*></p>)\s*(<p><strong>(?:图|表)\s*\d[^<]*</strong>[^<]*</p>)',
    r'<div class="figwrap">\1\2</div>', body)
# 单独出现的图片也居中
body = re.sub(r'(<p><img[^>]*></p>)', r'<div class="figwrap">\1</div>', body)

math_js = ''
if has_math:
    math_js = (
        '<script>window.MathJax={tex:{inlineMath:[[\'$\',\'$\']],displayMath:[[\'$$\',\'$$\']]},'
        'options:{skipHtmlTags:[\'script\',\'noscript\',\'style\',\'textarea\',\'pre\']}};</script>'
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>')

css = """
@page { size: A4; margin: 20mm 17mm 18mm 17mm; }
* { box-sizing: border-box; }
body {
  font-family: "Songti SC", "STSong", "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.7; color: #111;
  text-align: justify; margin: 0;
}
h1, h2, h3, h4 { font-family: "Heiti SC", "STHeiti", "PingFang SC", "Arial", sans-serif; }
h1 { font-size: 17pt; text-align: center; line-height: 1.4; margin: 0 0 4pt 0; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #bbb; padding-bottom: 3pt; margin: 20pt 0 8pt 0; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 6pt 0; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 4pt 0; }
p { margin: 5pt 0; }
strong { font-weight: 700; }
sup { font-size: 7.5pt; }
hr { border: none; border-top: 1px solid #ddd; margin: 14pt 0; }
img { max-width: 100%; height: auto; }
.figwrap { text-align: center; margin: 10pt 0 12pt 0; page-break-inside: avoid; }
.figwrap img { max-width: 94%; }
table {
  border-collapse: collapse; width: 100%; font-size: 9pt;
  margin: 8pt 0 10pt 0; page-break-inside: avoid; line-height: 1.5;
}
th, td { border: 0.7pt solid #999; padding: 3.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; font-weight: 700; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { margin: 6pt 0 6pt 10pt; padding-left: 8pt; border-left: 2.5pt solid #ccc; color: #333; }
pre {
  background: #f6f6f6; border: 0.6pt solid #ccc; border-radius: 3pt;
  padding: 8pt; font-family: "Menlo", "Consolas", monospace;
  font-size: 8pt; line-height: 1.4; white-space: pre; overflow-x: hidden;
  page-break-inside: avoid;
}
code { font-family: "Menlo", "Consolas", monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2pt; }
li { margin: 2pt 0; }
ol, ul { padding-left: 22pt; margin: 5pt 0; }
em { font-style: italic; }
a { color: #0645ad; text-decoration: none; }
.abstract { font-size: 10pt; margin: 8pt 24pt; }
.authors { text-align: center; font-size: 11pt; margin: 6pt 0; }
.affil { text-align: center; font-size: 9pt; color: #333; margin-bottom: 8pt; }
"""

# 作者/摘要区域的简单结构化(若 h1 后紧跟作者行)
body = body.replace('<p><strong>王莹</strong>', '<div class="authors"><p><strong>王莹</strong>', 1)
body = re.sub(r'(</p>\s*<p><sup>1</sup> 珠海学院[^<]*</p>)', r'\1</div>', body, count=1)
body = body.replace('<h2>摘要</h2>', '<h2>摘要</h2>\n<div class="abstract">', 1)
body = body.replace('<p><strong>关键词</strong>', '<p><strong>关键词</strong>')  # 无操作占位
body = body.replace('</p>\n\n<hr />\n\n<h1>', '</p></div>\n\n<hr />\n\n<h1>', 1) if False else body  # 不使用
body = re.sub(r'(<h2>摘要</h2>\s*<div class="abstract">.*?</p>)(\s*<p><strong>关键词)', r'\1</div>\2', body, flags=re.S)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>论文</title>
<style>{css}</style>
{math_js}
</head>
<body>
{body}
</body>
</html>"""

os.makedirs(BASE, exist_ok=True)
with open(HTML, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'HTML 已生成: {HTML} (公式: {"是,注入 MathJax" if has_math else "否"})')

def render_weasyprint(html_path, pdf_path):
    """主渲染后端: weasyprint (纯 Python, 精确 CSS 分页, 不执行 JS)。"""
    from weasyprint import HTML
    HTML(filename=html_path).write_pdf(pdf_path)


def render_chrome(html_path, pdf_path, has_math):
    """兜底渲染后端: Chrome headless (支持 MathJax JS 公式, 需已安装 Chrome)。"""
    CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(CHROME):
        raise RuntimeError('weasyprint 不可用且未找到 Chrome')
    prof = tempfile.mkdtemp(prefix='chrome_pdf_')
    cmd = [CHROME, '--headless=new', '--disable-gpu', '--no-sandbox',
           '--no-pdf-header-footer', f'--user-data-dir={prof}',
           f'--print-to-pdf={pdf_path}',
           f'--virtual-time-budget={60000 if has_math else 15000}',
           'file://' + html_path]
    print('运行 Chrome headless ...')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError('Chrome 错误: ' + r.stderr[-1500:])


# 渲染: 优先 weasyprint; 失败(如公式需 JS)自动回落 Chrome
try:
    print('渲染后端: weasyprint ...')
    render_weasyprint(HTML, PDF)
except Exception as e:
    print(f'weasyprint 失败({type(e).__name__}: {str(e)[:200]}), 尝试 Chrome 兜底')
    try:
        render_chrome(HTML, PDF, has_math)
    except Exception as e2:
        print('渲染失败:', e2)
        sys.exit(1)

if os.path.exists(PDF) and os.path.getsize(PDF) > 10000:
    print(f'PDF 已生成: {PDF} ({os.path.getsize(PDF)/1024:.0f} KB)')
else:
    print('警告: PDF 未生成或过小')
    sys.exit(1)
