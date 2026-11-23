#!/bin/bash
cd /home/user/Robot_learning/frontier
python3 arxiv_fresh.py; python3 fetch.py; python3 trending.py
python3 - <<'PY'
import json,urllib.request,urllib.parse,time,datetime as dt
ex=json.load(open('extra_repos.json')); have={e['repo'] for e in ex}
since=(dt.date.today()-dt.timedelta(days=3)).isoformat()
for q in ["humanoid","vision-language-action","legged locomotion","whole-body control","world action model"]:
    u="https://api.github.com/search/repositories?per_page=10&q="+urllib.parse.quote(f"{q} created:>{since}")
    try:
        for j in json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"fm"}),timeout=25))["items"]:
            if j["full_name"] not in have: ex.append(dict(title=j["description"] or j["name"],arxiv=None,repo=j["full_name"])); have.add(j["full_name"])
    except Exception as e: print("gh",e)
    time.sleep(2.5)
json.dump(ex,open('extra_repos.json','w'),indent=1)
PY
python3 verify.py > verify.log 2>&1
python3 - <<'PY'
import json,re
s=json.load(open('sent.json')); f=json.load(open('fresh.json'))['papers']; v=json.load(open('verified.json'))
kw=re.compile(r"humanoid|locomotion|legged|quadruped|whole-body|loco|world.action|vision-language-action|VLA|dexterous|motion tracking|teleop|robot",re.I)
np=[p for p in f if p['id'] not in s['papers'] and kw.search(p['title'])]
nr=[r for r in v if r['tier'] in 'ABR' and r['repo'] not in s['repos']]
out=dict(papers=np,repos=nr); json.dump(out,open('new_items.json','w'),indent=1)
print(len(np),'new papers',len(nr),'new verified repos')
PY
echo CYCLE_DONE
