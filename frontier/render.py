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
import re as _re
_bad=_re.compile(r"arxiv|daily|awesome|paper|list|watch|digest|weekly|survey|handbook|notes|github\.io|profile|^[^/]+/[^/]+$" if False else r"arxiv|daily|awesome|paper|list|watch|digest|weekly|survey|handbook|notes|github\.io",_re.I)
R=json.load(open('/home/user/Robot_learning/frontier/repro.json'))
rows_=""
for p in R:
    rs=[r for r in p["repos"] if not _bad.search(r["repo"].split("/")[1]) and r["repo"].split("/")[0].lower()!=r["repo"].split("/")[1].lower() and r["stars"]+len(r["desc"])>0]
    if rs: rows_+=f'<tr><td class="n">{p["date"]}</td><td>{E(p["title"])}<div class="s">'+" · ".join(f'<a href="https://github.com/{r["repo"]}">{r["repo"]}</a> ★{r["stars"]} ({r["kind"]})' for r in rs[:3])+"</div></td></tr>"
body+='<h2>Code found on GitHub for recent papers</h2><p class="m">Repos whose README cites the paper. Papers under 2 weeks old rarely have independent reproductions yet; &ldquo;mentions&rdquo; means the repo cites the paper and is not confirmed official.</p><div class="w"><table>'+rows_+'</table></div>'
body+='<h2>Community reproductions found (older papers)</h2><div class="w"><table><tr><td><a href="https://github.com/Motphys/Uni-SuperMimic">Motphys/Uni-SuperMimic</a> ★314<div class="s">independent reproduction of SuperMimic on UniLab</div></td></tr><tr><td><a href="https://github.com/bytepioneerX/Acp_supermimic_vibed">bytepioneerX/Acp_supermimic_vibed</a><div class="s">ACP reproduction + SuperMimic for Unitree G1</div></td></tr><tr><td><a href="https://github.com/kaibohex/muscle-supermimic">kaibohex/muscle-supermimic</a><div class="s">SuperMimic tracking for muscle-driven humanoids on mjlab</div></td></tr><tr><td><a href="https://github.com/EGalahad/mimic-lite">EGalahad/mimic-lite</a> ★76<div class="s">efficient humanoid motion tracking, scalable sim-to-real training</div></td></tr></table></div>'
F=json.load(open('/home/user/Robot_learning/frontier/fresh.json'))
V={ (v["arxiv"] or v["title"], v["repo"]):v for v in json.load(open('/home/user/Robot_learning/frontier/verified.json'))}
def fr(p):
    h=p.get("hf"); extra=""
    if h: extra=f' · HF ▲{h["up"]}'+(f' · <a href="{E(h["repo"])}">repo</a>' if h.get("repo") else "")
    return f'<tr><td class="n">{p["date"]}</td><td><a href="https://arxiv.org/abs/{p["id"]}">{E(p["title"])}</a><div class="s">{E(", ".join(p["authors"]))} · {p["cat"]}{extra}</div></td></tr>'
fresh='<h2>Fresh from arXiv (last 3 days, direct feed)</h2><div class="w"><table>'+"".join(fr(p) for p in F["papers"][:60])+'</table></div>'
TIER={"A":"released, official claim","B":"released, paper cited","R":"independent reproduction"}
vr=[v for v in json.load(open('/home/user/Robot_learning/frontier/verified.json')) if v["tier"] in TIER]
seen=set(); vrows=""
for v in sorted(vr,key=lambda x:x["tier"]):
    if v["repo"] in seen: continue
    seen.add(v["repo"]); vrows+=f'<tr><td>{v["tier"]}</td><td><a href="https://github.com/{v["repo"]}">{v["repo"]}</a><div class="s">{E(v["title"][:80])} · {TIER[v["tier"]]}{" · weights" if v.get("weights") else ""}</div></td></tr>'
nr=sum(1 for v in json.load(open('/home/user/Robot_learning/frontier/verified.json')) if v["tier"]=="S")
ver=f'<h2>Verified code (README read)</h2><p class="m">Tier A/B/R only. {nr} repos were excluded because the README says code is not released or the repo is only a page.</p><div class="w"><table>{vrows}</table></div>'
body=fresh+ver+body
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
