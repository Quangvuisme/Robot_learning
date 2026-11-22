import json,html,sys
d=json.load(open('/home/user/Robot_learning/frontier/data.json'))
E=lambda s:html.escape(s or "")
secs={}
for i in d['items']: secs.setdefault(i['section'],[]).append(i)
order=["Locomotion","Loco-Manipulation and Whole-Body-Control","Manipulation","Teleoperation","Sim-to-Real","Navigation","State Estimation","Simulation Benchmark","Hardware Design"]
def row(i):
    l=[f'<a href="{E(i["url"])}">{E(i["venue"])}</a>']
    for k in("website","code","video"):
        if i.get(k): l.append(f'<a href="{E(i[k])}">{k}</a>')
    g=i.get("gh"); extra=""
    if g:
        tag="official" if g["official"] else "unverified match"
        extra=f' · <a href="https://github.com/{g["full"]}">{g["full"]}</a> ★{g["stars"]} ({g["spd"]}/day), pushed {g["pushed"]} [{tag}]'
    return f'<tr><td class="n">{i["date"]}</td><td>{E(i["title"])}<div class="s">{" · ".join(l)}{extra}</div></td></tr>'
body=""
for s in order+[k for k in secs if k not in order]:
    if s in secs: body+=f'<h2>{E(s)} <span class="m">({len(secs[s])})</span></h2><div class="w"><table>{"".join(row(i) for i in secs[s])}</table></div>'
T=json.load(open('/home/user/Robot_learning/frontier/trending.json'))
body+='<h2>GitHub buzz (repos pushed in last 30 days, sorted by stars)</h2><div class="w"><table>'+"".join(f'<tr><td class="n">★{t["stars"]}</td><td><a href="https://github.com/{t["repo"]}">{t["repo"]}</a><div class="s">pushed {t["pushed"]} · {E(t["desc"])}</div></td></tr>' for t in T)+'</table></div>'
nc=sum('code' in i for i in d['items'])
open('/tmp/claude-0/-home-user-Robot-learning/0673c822-7c37-5a1e-9ffa-49cb001bb9d2/scratchpad/frontier.html','w').write(f'''<title>Humanoid Frontier Map</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
:root{{--bg:#f6f7f8;--fg:#14181c;--mut:#5a6570;--line:#d9dee3;--acc:#0b6e99;--card:#fff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1316;--fg:#e6eaed;--mut:#8f9aa4;--line:#27303a;--acc:#5ab8e0;--card:#161c21;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0f1316;--fg:#e6eaed;--mut:#8f9aa4;--line:#27303a;--acc:#5ab8e0;--card:#161c21;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding-block:28px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
main{{max-width:960px;margin:auto}} h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:18px;margin:30px 0 8px}}
a{{color:var(--acc)}} .m{{color:var(--mut);font-weight:400}} .w{{overflow-x:auto;border:1px solid var(--line);background:var(--card);border-radius:6px}}
table{{border-collapse:collapse;width:100%}} td{{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}} tr:last-child td{{border:0}}
.s{{font-size:12.5px;color:var(--mut)}} .n{{font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--mut);font-size:13px}}
</style>
<main><h1>Humanoid &amp; Robot Learning Frontier Map</h1>
<p class="m">Updated {d["generated"]} · {len(d["items"])} papers from the last 45 days in YanjieZe/awesome-humanoid-robot-learning · {nc} list entries link official code, the rest link a project website or only the paper. Repo stats come from the GitHub search API; matches marked unverified are guesses by name.</p>{body}</main>''')
