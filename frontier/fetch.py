#!/usr/bin/env python3
"""Pull latest papers from YanjieZe/awesome-humanoid-robot-learning, attach official code
links, enrich with GitHub repo stats (stars, forks, age, last push), write data.json."""
import re, json, sys, time, datetime as dt, urllib.request, os
HERE=os.path.dirname(os.path.abspath(__file__))
RAW="https://raw.githubusercontent.com/YanjieZe/awesome-humanoid-robot-learning/main/README.md"
DAYS=int(os.environ.get("DAYS","45")); MAXCALLS=int(os.environ.get("MAXCALLS","50"))
def get(u):
    r=urllib.request.Request(u,headers={"User-Agent":"frontier-map"})
    return urllib.request.urlopen(r,timeout=25).read().decode()
md=get(RAW)
today=dt.date.today()
sec=None; items=[]
for ln in md.splitlines():
    if ln.startswith("## "): sec=ln[3:].strip(); continue
    m=re.match(r"- \[(arXiv|CoRL|RSS|ICRA|IROS|CVPR|ICLR|NeurIPS|ICML|RA-L|Science|Nature|TRO|[A-Za-z\-]+) ?(\d{4})\.(\d{1,2})(?:\.(\d{1,2}))?\]\((.+?)\),\s*(.*)",ln)
    if not m or not sec: continue
    venue,y,mo,d,url,rest=m.groups()
    date=dt.date(int(y),int(mo),int(d or 1))
    if (today-date).days>DAYS: continue
    title=re.split(r",\s*\[(?:website|code|paper|video|demo)",rest)[0].strip()
    links=dict((k.lower(),v) for k,v in re.findall(r"\[(website|code|paper|video|demo)\]\((.+?)\)",rest))
    items.append(dict(section=sec,venue=venue,date=str(date),url=url,title=title,**links))
# GitHub enrichment via search API (user-approved). 30 req/min unauthenticated.
import urllib.parse
cache_f=os.path.join(HERE,"ghcache.json")
cache=json.load(open(cache_f)) if os.path.exists(cache_f) else {}
def ent_of(j):
    age=max(1,(today-dt.date.fromisoformat(j["created_at"][:10])).days)
    return dict(t=time.time(),stars=j["stargazers_count"],forks=j["forks_count"],issues=j["open_issues_count"],
        pushed=j["pushed_at"][:10],created=j["created_at"][:10],spd=round(j["stargazers_count"]/age,2),
        license=(j.get("license") or {}).get("spdx_id"),full=j["full_name"])
def search(q):
    j=json.loads(get("https://api.github.com/search/repositories?per_page=10&q="+urllib.parse.quote(q))); time.sleep(2.2)
    return j.get("items",[])
repos={}
for it in items:
    m=re.match(r"https://github\.com/([^/]+)/([^/#?]+)",it.get("code",""))
    if m: repos[f"{m[1]}/{m[2]}".removesuffix(".git").lower()]=it
need=[k for k in repos if k not in cache or time.time()-cache[k]["t"]>3*3600]
for i in range(0,len(need),5):
    try:
        for j in search(" ".join("repo:"+k for k in need[i:i+5])): cache[j["full_name"].lower()]=ent_of(j)
    except Exception as e: print("gh fail",e,file=sys.stderr)
for k,it in repos.items():
    if k in cache: it["gh"]=dict(official=True,**{x:y for x,y in cache[k].items() if x!="t"})
tried=0
for it in items:
    if "gh" in it or "code" in it or tried>=20: continue
    name=re.split(r"[:\-]",it["title"])[0].strip()
    if len(name)<4 or " " in name: continue
    key="cand:"+name.lower()
    if key not in cache or time.time()-cache[key]["t"]>6*3600:
        tried+=1
        try:
            hit=[j for j in search(f"{name} humanoid in:name,description,readme") if name.lower() in j["name"].lower()]
            cache[key]=dict(t=time.time(),hit=ent_of(hit[0]) if hit else None)
        except Exception as e: print("cand fail",name,e,file=sys.stderr); continue
    h=cache[key].get("hit")
    if h: it["gh"]=dict(official=False,**{x:y for x,y in h.items() if x!="t"})
json.dump(cache,open(cache_f,"w"))
items.sort(key=lambda x:(x["date"],x.get("gh",{}).get("stars",0)),reverse=True)
json.dump(dict(generated=str(dt.datetime.now(dt.timezone.utc))[:16]+"Z",items=items),open(os.path.join(HERE,"data.json"),"w"),indent=1,ensure_ascii=False)
print(len(items),"papers;",sum(1 for i in items if "gh" in i),"with GH stats;")
