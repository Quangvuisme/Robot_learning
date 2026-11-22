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
# GitHub enrichment (unauthenticated API, 60 req/h; cache to stay under)
cache_f=os.path.join(HERE,"ghcache.json")
cache=json.load(open(cache_f)) if os.path.exists(cache_f) else {}
calls=0
for it in items:
    c=it.get("code","")
    m=re.match(r"https://github\.com/([^/]+)/([^/#?]+)",c)
    if not m: continue
    key=f"{m[1]}/{m[2]}".removesuffix(".git")
    ent=cache.get(key)
    if (not ent or time.time()-ent["t"]>3*3600) and calls<MAXCALLS:
        try:
            j=json.loads(get(f"https://api.github.com/repos/{key}")); calls+=1
            age=max(1,(today-dt.date.fromisoformat(j["created_at"][:10])).days)
            ent=dict(t=time.time(),stars=j["stargazers_count"],forks=j["forks_count"],issues=j["open_issues_count"],
                     pushed=j["pushed_at"][:10],created=j["created_at"][:10],lang=j.get("language"),
                     spd=round(j["stargazers_count"]/age,2),license=(j.get("license") or {}).get("spdx_id"))
            cache[key]=ent
        except Exception as e: print("gh fail",key,e,file=sys.stderr)
    if ent: it["gh"]=dict(repo=key,**{k:v for k,v in ent.items() if k!="t"})
json.dump(cache,open(cache_f,"w"))
items.sort(key=lambda x:(x["date"],x.get("gh",{}).get("stars",0)),reverse=True)
json.dump(dict(generated=str(dt.datetime.utcnow())[:16]+"Z",items=items),open(os.path.join(HERE,"data.json"),"w"),indent=1,ensure_ascii=False)
print(len(items),"papers;",sum(1 for i in items if "gh" in i),"with GH stats;",calls,"api calls")
