#!/usr/bin/env python3
"""For each paper find GitHub repos by others: official code, unofficial reimplementations/reproductions.
Signals: repos whose README mentions the arXiv id, and name+keyword searches. Search API, ~30 req/min."""
import json,re,time,urllib.request,urllib.parse,os
HERE=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(HERE,"data.json")))
cf=os.path.join(HERE,"repro_cache.json"); C=json.load(open(cf)) if os.path.exists(cf) else {}
def S(q):
    u="https://api.github.com/search/repositories?per_page=8&sort=stars&order=desc&q="+urllib.parse.quote(q)
    for _ in range(2):
        try: r=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"fm"}),timeout=25)); time.sleep(2.4); return r.get("items",[])
        except Exception as e: time.sleep(20)
    return []
KW=re.compile(r"reproduc|unofficial|re-?implement|replicat|independent|port of|clone",re.I)
out=[]
for it in d["items"]:
    aid=it["url"].rsplit("/",1)[-1] if "arxiv.org" in it["url"] else None
    name=re.split(r"[:\-]",it["title"])[0].strip()
    key=aid or it["title"]
    if key in C and time.time()-C[key]["t"]<6*3600: res=C[key]["r"]
    else:
        found={}
        qs=[]
        if aid: qs.append(f'"{aid}" in:readme')
        if name and " " not in name and len(name)>3: qs.append(f'{name} (reproduction OR unofficial OR reimplementation OR "official implementation") in:name,description,readme')
        for q in qs:
            for j in S(q):
                txt=f'{j["name"]} {j.get("description") or ""}'
                if aid is None and name.lower() not in j["name"].lower() and name.lower() not in txt.lower(): continue
                kind="reproduction" if KW.search(txt) else ("official?" if re.search(r"official",txt,re.I) else "mentions")
                found[j["full_name"]]=dict(repo=j["full_name"],stars=j["stargazers_count"],pushed=j["pushed_at"][:10],kind=kind,desc=(j.get("description") or "")[:100])
        res=sorted(found.values(),key=lambda x:-x["stars"])[:5]; C[key]=dict(t=time.time(),r=res); json.dump(C,open(cf,"w"))
    if res: out.append(dict(title=it["title"],date=it["date"],url=it["url"],repos=res))
json.dump(out,open(os.path.join(HERE,"repro.json"),"w"),indent=1,ensure_ascii=False)
print(len(out),"papers with community/official repos")
