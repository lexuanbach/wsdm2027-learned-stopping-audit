"""Prospectively frozen LS13: fresh question-disjoint 8-search trajectories.
No model output is generated until protocol.json and selected IDs are saved.
Reuses the submitted answer/query prompts, BM25 corpus and feature definitions.
The $25 cap includes conservative reservations for all concurrent requests.
"""
from pathlib import Path
import os,sys,json,hashlib,threading,time,sqlite3,re,concurrent.futures,pickle
import numpy as np
import pandas as pd
import anthropic
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import run_experiment as rx
# Completed trajectories in OUT are reused. Set LONG_HORIZON_OUT to an empty
# folder to collect a new set instead of reusing the archived one.
OUT=Path(os.environ.get('LONG_HORIZON_OUT',ROOT/'results/long_horizon'));OUT.mkdir(parents=True,exist_ok=True)
MODEL='claude-haiku-4-5-20251001';CAP=25.; lock=threading.Lock();local=threading.local()
ledger={'estimated_usd':0.000035,'reserved_usd':0.,'input_tokens':10,'output_tokens':5,'requests':1,'failed_requests':0,'note':'List-price estimate, not invoice; includes initial 10-input/5-output connectivity probe.'}
if (OUT/'spend.json').exists():ledger=json.loads((OUT/'spend.json').read_text())
def save_ledger():
 p=OUT/'spend.tmp';p.write_text(json.dumps(ledger,indent=2));p.replace(OUT/'spend.json')
def call(system,user,max_tokens):
 if not hasattr(local,'client'):local.client=anthropic.Anthropic(timeout=60,max_retries=0)
 # A UTF-8 byte bound plus a 1024-token framing allowance exceeds this prompt's token count.
 reservation=(len((system+user).encode())+1024+5*max_tokens)/1e6
 with lock:
  if ledger['estimated_usd']+ledger['reserved_usd']+reservation>CAP:raise RuntimeError('USD25 budget cap reached')
  ledger['reserved_usd']+=reservation;save_ledger()
 started=time.perf_counter()
 try:
  r=local.client.messages.create(model=MODEL,max_tokens=max_tokens,temperature=0.,system=system,messages=[{'role':'user','content':user}])
 except Exception as e:
  with lock:
   ledger['failed_requests']+=1
   # Keep conservative reservation: response may have been lost after generation.
   save_ledger()
  raise RuntimeError(f'API failed: {type(e).__name__}; status {getattr(e,"status_code",None)}') from None
 cost=(r.usage.input_tokens+5*r.usage.output_tokens)/1e6
 with lock:
  ledger['reserved_usd']-=reservation;ledger['estimated_usd']+=cost
  ledger['input_tokens']+=r.usage.input_tokens;ledger['output_tokens']+=r.usage.output_tokens;ledger['requests']+=1;save_ledger()
 text=''.join(b.text for b in r.content if b.type=='text').strip()
 return text,{'in':r.usage.input_tokens,'out':r.usage.output_tokens,'calls':1,'latency_seconds':time.perf_counter()-started}
class Retriever:
 def search(self,query,k):
  if not hasattr(local,'db'):local.db=sqlite3.connect(f'file:{OUT}/corpus.sqlite?mode=ro',uri=True)
  toks=rx._tok(query)
  if not toks:return []
  match=' OR '.join(f'"{t}"' for t in toks)
  return [r[0] for r in local.db.execute('SELECT text FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',(match,int(k)))]
def collect(task):
 split,i,row=task;p=OUT/'trajectories'/f'{split}-{i:04d}.json'
 if p.exists():return split,i,'cached'
 env=rx.QuestionEnv(row);env.sents=[];env.bm25=Retriever();ev=[];traj=[]
 for step in range(9):
  qu={};lat=0.;q=None
  if step:
   known='\n'.join(f'- {e}' for e in ev) if ev else '(nothing yet)'
   raw,qu=call('You issue one short search query to find missing facts for a multi-hop question. Output only the query.',f'Question: {env.question}\nKnown so far:\n{known}\nNext search query:',32)
   lines=[s for s in raw.splitlines() if s.strip()];q=re.sub(r'^["\']|["\']$','',lines[0] if lines else '').strip() or env.question
   t=time.perf_counter();hits=env.search(q,k=3);lat=time.perf_counter()-t
   for h in hits:
    if h not in ev:ev.append(h)
  evidence='\n'.join(f'- {e}' for e in ev) if ev else '(no evidence)'
  answer,au=call(rx.SYS,f'Evidence:\n{evidence}\n\nQuestion: {env.question}\nRespond with exactly: Answer: <short answer>',40)
  item={'searches':step,'evidence':list(ev),'answer':rx.extract_answer(answer),'answer_usage':au,'retrieval_latency_seconds':lat}
  if step:item.update(query=q,query_usage=qu)
  traj.append(item)
 record={'id':env.qid,'question':env.question,'gold_answers':[env.answer]+env.aliases,'support_titles':env.support_titles,'support_sentences':env.support_sentences,'trajectory':traj,'prefix_em':[env.em(p['answer']) for p in traj],'prefix_f1':[env.f1(p['answer']) for p in traj],'prefix_features':[rx.prefix_features(env,traj,t) for t in range(9)]}
 p.write_text(json.dumps(record,indent=1));return split,i,'new'
def main():
 data=ROOT/'data/hotpot_val.parquet';df=pd.read_parquet(data)
 old=set();question_ids={str(r['question']):str(r['id']) for _,r in df.iterrows()}
 for p in (ROOT/'results/measured').glob('run_*.json'):
  for r in json.loads(p.read_text()).get('per_q',[]):old.add(str(r.get('id',r.get('qid',question_ids.get(r.get('question',''),'unavailable')))))
 for p in (ROOT/'results/measured').glob('critic_data*.pkl'):
  for r in pickle.loads(p.read_bytes()).get('traces',[]):old.add(str(r.get('id',r.get('qid',question_ids.get(r.get('question',''),'unavailable')))))
 avail=df[~df['id'].astype(str).isin(old)].sample(frac=1.,random_state=212613)
 assert len(avail)>=320,len(avail)
 rows=[row for _,row in avail.iloc[:320].iterrows()]
 protocol={'date':'2026-11-18','model':MODEL,'temperature':0,'search_horizon':8,'train':120,'test':200,'seed':212613,'excluded_existing_ids':len(old),'eligible_questions':len(avail),'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'selection':'First 120 train, next 200 test after excluding every archived training/test question ID; train-only cross-fitting and frozen threshold selection; replay horizons 4 and 8 on the same fresh adaptive-query trajectories.','inference':'Report all fixed budgets, repeat, FBG-only, guarded FBG, evidence growth; paired question-level uncertainty. No claim of a jointly learned query policy.','budget_usd':25,'pricing':'https://platform.claude.com/docs/en/about-claude/pricing; Haiku 4.5: USD1/M input, USD5/M output','ids':{'train':[str(r['id']) for r in rows[:120]],'test':[str(r['id']) for r in rows[120:]]}}
 pp=OUT/'protocol.json'
 if pp.exists():assert json.loads(pp.read_text())==protocol
 else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
 (OUT/'trajectories').mkdir(exist_ok=True)
 if not (OUT/'corpus.sqlite').exists():
  pool=list(dict.fromkeys(s for _,r in df.iterrows() for s in rx.row_sentences(r)))
  index=rx.CorpusBM25(pool);db=sqlite3.connect(OUT/'corpus.sqlite');index.db.backup(db);db.close();print('Indexed',len(pool),'units',flush=True)
 tasks=[('train' if i<120 else 'test',i if i<120 else i-120,row) for i,row in enumerate(rows)]
 print('Frozen protocol:',len(old),'excluded IDs,',len(avail),'eligible, 320 new trajectories',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  fs=[ex.submit(collect,t) for t in tasks]
  for n,f in enumerate(concurrent.futures.as_completed(fs),1):
   try:result=f.result()
   except Exception:
    for pending in fs:pending.cancel()
    raise
   if n%10==0:print(n,'/320 completed; USD',round(ledger['estimated_usd'],4),flush=True)
 print('COMPLETE',json.dumps(ledger),flush=True)
if __name__=='__main__':main()
