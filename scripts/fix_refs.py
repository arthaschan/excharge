#!/usr/bin/env python3
"""修正参考文献编号: 当前编号 1-8 做循环移位(1->3,2->4,...,6->8,7->1,8->2), 9-16 不变。
背景: 之前的重排基于错误的"首次出现顺序"(漏了正文第一处 [14,2])。正确顺序应让
MTAD-GAT/GDN 排最前。本脚本一次性修正正文引用 + 参考文献列表。
"""
import re

path = 'paper/paper_draft.md'
text = open(path).read()

# 当前(错误)编号 -> 正确编号: 1->3, 2->4, 3->5, 4->6, 5->7, 6->8, 7->1, 8->2, 9-16 不变
def rot(n):
    if 1 <= n <= 8:
        return ((n + 1) % 8) + 1
    return n

# 1. 分离正文与参考文献
idx = text.find('## 参考文献')
body = text[:idx]
refs = text[idx:]

# 2. 正文引用重编号
def repl(m):
    return f'[{rot(int(m.group(1)))}]'
body_new = re.sub(r'\[(\d+)\]', repl, body)

# 3. 解析参考文献列表(每条目一行, 以 [N] 开头; 跳过 --- 和 *Manuscript* 等非条目行)
entries = {}
for line in refs.split('\n'):
    m = re.match(r'^\[(\d+)\]\s', line)
    if m:
        entries[int(m.group(1))] = line

assert len(entries) == 16, f'解析到 {len(entries)} 个条目, 应为 16'

# 4. 正确顺序: 目标[1..16] = 当前[7,8,1,2,3,4,5,6,9,10,11,12,13,14,15,16]
target_order = [7, 8, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
new_lines = []
for new_n, cur_n in enumerate(target_order, 1):
    line = entries[cur_n]
    new_line = re.sub(r'^\[\d+\]', f'[{new_n}]', line)
    new_lines.append(new_line)

refs_new = '## 参考文献\n\n' + '\n\n'.join(new_lines) + '\n'

# 5. 写回
out = body_new + refs_new + '\n---\n\n*Manuscript prepared for TAIG 2026 - Technologies for AI Governance*\n'
open(path, 'w').write(out)
print('修正完成。正确顺序:')
for new_n, cur_n in enumerate(target_order, 1):
    print(f'  [{new_n}] = 原[{cur_n}] {new_lines[new_n-1][:60]}')
