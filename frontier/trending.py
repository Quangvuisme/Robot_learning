import json,urllib.request,urllib.parse,time,datetime as dt
since=(dt.date.today()-dt.timedelta(days=30)).isoformat()
Q=["humanoid whole-body control","humanoid motion tracking","humanoid locomotion reinforcement learning","vision-language-action","world action model robot","legged locomotion sim-to-real","humanoid teleoperation","dexterous manipulation"]
out={}
for q in Q:
    u="https://api.github.com/search/repositories?per_page=6&sort=stars&order=desc&q="+urllib.parse.quote(f"{q} pushed:>{since} stars:>40")
    try:
        for j in json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"fm"}),timeout=25)).get("items",[]):
            out[j["full_name"]]=dict(repo=j["full_name"],stars=j["stargazers_count"],pushed=j["pushed_at"][:10],created=j["created_at"][:10],desc=(j["description"] or "")[:110],q=q)
    except Exception as e: print("fail",q,e)
    time.sleep(2.5)
json.dump(sorted(out.values(),key=lambda x:-x["stars"]),open("trending.json","w"),indent=1,ensure_ascii=False)
print(len(out))
