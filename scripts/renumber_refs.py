#!/usr/bin/env python3
"""重排论文参考文献编号(按正文首次出现顺序), 回应用 #10 引用编号顺序乱。"""
import re

path = 'paper/paper_draft.md'
text = open(path).read()

# 旧编号 -> 新编号 (按正文首次出现顺序)
mapping = {3: 1, 11: 2, 4: 3, 6: 4, 1: 5, 15: 6, 14: 7, 2: 8,
           5: 9, 16: 10, 12: 11, 13: 12, 9: 13, 8: 14, 7: 15, 10: 16}
# 新编号 -> 旧编号
inv = {v: k for k, v in mapping.items()}

# 1. 分离正文与参考文献列表
ref_marker = '## 参考文献'
idx = text.find(ref_marker)
assert idx != -1, '找不到参考文献标记'
body = text[:idx]
refs = text[idx:]

# 2. 重编号正文引用 (处理 [14, 2] 这类多引用)
def repl(m):
    return f'[{mapping[int(m.group(1))]}]'
body_new = re.sub(r'\[(\d+)\]', repl, body)

# 3. 解析参考文献列表(每条目一行, 以 [N] 开头)
ref_lines = refs.split('\n')
entries = {}  # 旧编号 -> 条目文本(含 [N] 前缀)
cur = None
for line in ref_lines:
    m = re.match(r'^\[(\d+)\]', line)
    if m:
        cur = int(m.group(1))
        entries[cur] = line
    elif cur is not None and line.strip():
        # 续行(当前无多行条目, 但稳妥处理)
        entries[cur] = entries.get(cur, '') + ' ' + line.strip()

# 4. 按新编号顺序重建列表
new_list = []
for new_n in range(1, 17):
    old_n = inv[new_n]
    line = entries[old_n]
    # 把 [old_n] 前缀替换为 [new_n]
    new_line = re.sub(r'^\[\d+\]', f'[{new_n}]', line)
    new_list.append(new_line)

# 重建参考文献段落
refs_new = '## 参考文献\n\n' + '\n\n'.join(new_list) + '\n'

# 5. 写回
out = body_new + refs_new
# 保留尾部 (--- 和 Manuscript prepared)
tail = text[text.rfind('\n---\n'):]
out = body_new + refs_new + '\n---\n\n*Manuscript prepared for TAIG 2026 - Technologies for AI Governance*\n'

open(path, 'w').write(out)
print('重排完成。新编号顺序:')
for new_n in range(1, 17):
    old_n = inv[new_n]
    first = new_list[new_n-1].split('.')[0][:60]
    print(f'  [{new_n}] <- 旧[{old_n}] {first}...')
