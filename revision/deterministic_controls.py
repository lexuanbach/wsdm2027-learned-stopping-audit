from pathlib import Path
import json,sys,time,collections,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import round9_analyses as r9

def stops_for(feats, threshold):
 return np.array([next((t for t in range(1,len(f)) if f[t,3]<=threshold),len(f)-1) for f in feats])

def main():
 audit=json.loads((ROOT/'revision/numerical-audit.json').read_text())
 thresholds={r['tag']:r['evidence_growth'] for r in audit['retrospective_analyses']}
 rows=[]; vectors=[]; ids=[]
 for spec in r9.fresh_cells():
  c=r9.Cell(*spec); selected=thresholds[c.tag]; th=selected['threshold']
  # Selection was already frozen on the training chains by analyze_revision.py.
  train_ids={str(c.traces[i]['id']) for i in c.fit_ids}
  assert not train_ids & {str(r['id']) for r in c.records}
  st=stops_for(c.feats,th); start=time.perf_counter()
  for _ in range(100): stops_for(c.feats,th)
  overhead=(time.perf_counter()-start)/(100*c.n)
  eg=c.point(st); eg['latency']+=overhead
  assert abs(eg['em']-selected['em'])<1e-9
  simple={**c.simple_points(),'evidence_growth':eg}
  sn={k:(p['em'],p['calls']) for k,p in simple.items()}
  row={'tag':c.tag,'n_test':c.n,'threshold':th,'training_grid':selected['training_grid'],
       'control_cpu_seconds_per_question':overhead,'extra_llm_calls':0,'extra_llm_tokens':0,'policies':{}}
  for name,stop in [('guarded',c.g_stop),('value',c.v_stop)]:
   p=c.point(stop); v,dom=r9.verdict_eps(p['em'],p['calls'],sn,f'fixed_{c.bstar}',0.)
   row['policies'][name]={'category':{'W':'I','T':'F','L':'D'}[v],'dominated_by':dom,'contrasts':{}}
   for coord in ['calls','tokens','latency']:
    pts={k:{**z,'cost':z[coord]} for k,z in simple.items()}
    hull=r9.upper_hull(pts); mix,n1,n2,w=r9.hull_mixture(hull,p[coord],pts)
    diff=p['em_vec']-mix
    out={'mean_delta_em':100*float(diff.mean()),'mixture':[n1,n2],'weight_first':float(w),'paired_ci':r9.paired_ci(diff)}
    row['policies'][name]['contrasts'][coord]=out
    vectors.append(diff);ids.append([str(r['id']) for r in c.records])
  rows.append(row)
 # Gaussian multiplier bands, sharing question weights across cells and costs.
 # Observed mixture vertices/weights held fixed: explicitly conditional inference.
 qids=sorted(set(q for cellids in ids for q in cellids)); qi={q:i for i,q in enumerate(qids)}
 influence=np.zeros((len(qids),len(vectors)))
 for j,(d,q) in enumerate(zip(vectors,ids)):
  for v,k in zip((d-d.mean())/np.sqrt(len(d)*(len(d)-1)),q): influence[qi[k],j]+=v
 se=np.sqrt((influence**2).sum(axis=0)); valid=se>0
 rng=np.random.default_rng(212612); mx=[]
 for _ in range(100):
  z=rng.standard_normal((100,len(qids)))@influence
  mx.extend(np.max(np.abs(z[:,valid]/se[valid]),axis=1))
 crit=float(np.quantile(mx,.95)); k=0
 for r in rows:
  for name in ['guarded','value']:
   for coord in ['calls','tokens','latency']:
    o=r['policies'][name]['contrasts'][coord]; m=o['mean_delta_em'];o['conditional_simultaneous_ci95']=[m-100*crit*se[k],m+100*crit*se[k]]
    o['conditional_verdict']='win' if o['conditional_simultaneous_ci95'][0]>0 else 'loss' if o['conditional_simultaneous_ci95'][1]<0 else 'unresolved';k+=1
 summary={}
 for name in ['guarded','value']:
  summary[name]={'DFI':dict(collections.Counter(r['policies'][name]['category'] for r in rows)), 'costs':{}}
  for coord in ['calls','tokens','latency']:
   cs=[r['policies'][name]['contrasts'][coord] for r in rows]
   summary[name]['costs'][coord]={'mean_delta_em':float(np.mean([z['mean_delta_em'] for z in cs])), 'conditional_verdicts':dict(collections.Counter(z['conditional_verdict'] for z in cs))}
 out={'protocol':'Retrospective training-selected evidence-growth control; 14 cells, two policies, three cost coordinates. Existing fixed/repetition references extended by one control.',
 'inference':'10,000 Gaussian multiplier replicates; shared question-ID weights across cells; simultaneous 84-contrast max-absolute-t family. Conditional on fitted policies and observed hull vertices/weights; does not cover frontier reselection or training uncertainty. Not confirmatory.',
 'wall_time':'Historical logged cumulative LLM+retrieval latency plus local measured control decision overhead (100 repeats). Mixed-hardware sensitivity, not a fresh end-to-end wall-time benchmark.',
 'machine':platform.machine(),'critical_value':crit,'n_contrasts':len(vectors),'cells':rows,'summary':summary}
 (ROOT/'revision/deterministic-controls.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=[]
 for name,label in [('guarded','Guarded'),('value','FBG-only')]:
  for coord,labelc in [('calls','Calls'),('tokens','Input tokens'),('latency','Wall-time proxy')]:
   s=summary[name]['costs'][coord];v=s['conditional_verdicts']
   lines.append(f"{label} & {labelc} & ${s['mean_delta_em']:+.2f}$ & {v.get('win',0)}/{v.get('unresolved',0)}/{v.get('loss',0)} \\\\")
 (ROOT/'manuscript/tables/deterministic_controls_rows.tex').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'summary':summary,'critical_value':crit,'overhead_range':[min(r['control_cpu_seconds_per_question'] for r in rows),max(r['control_cpu_seconds_per_question'] for r in rows)]},indent=2))
if __name__=='__main__':main()
