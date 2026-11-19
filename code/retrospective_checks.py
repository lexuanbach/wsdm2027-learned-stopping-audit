"""Retrospective offline audits; fixed formulas/grids, no endpoint calls."""
from pathlib import Path
import sys,json,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import round9_analyses as r9
out={};rnd=json.loads((ROOT/'results/measured/round9_analyses.json').read_text())
rows=[[x.strip() for x in line.split('&')] for line in (ROOT/'outputs/tables/decision_matrix_rows.tex').read_text().splitlines() if '&' in line]
out['table_core_counts']={p:dict(collections.Counter(c[ix] for c in rows)) for p,ix in [('value',7),('guarded',10)]}
out['table_instrumented_counts']=dict(collections.Counter(c[10] for c in rows[-14:]))
out['refit_counts']=rnd['summary']['joint_ablation'];out['logged_target_counts']=rnd['summary']['target_variants']['fbg_guarded']
out['refit_threshold_reproduced_cells']=rnd['summary']['fbg_refit_reproduces_logged_lambda_cells']
for p,ix in [('value_only',5),('guarded',8)]:out[p+'_savings_pct_from_table_calls']=[100*(1-(float(c[ix].split('/')[1])-1)/8) for c in rows[:9]]
ext=json.loads((ROOT/'results/round10/extended_decision_matrix.json').read_text())
for policy in ['value_only','guarded_value']:
 entries=[]
 for c in ext['cells']:
  m=c['policies'].get(policy,{}).get('vs_matched_mixture_on_calls')
  if m:
   lo,hi=m['delta_em']['ci95'];entries.append({'cell':c.get('tag',c.get('cell')),'width_em':hi-lo,'ci95':[lo,hi]})
 out['precision_'+policy]={'n':len(entries),'width_min':min(e['width_em'] for e in entries),'width_median':float(np.median([e['width_em'] for e in entries])),'width_max':max(e['width_em'] for e in entries),'entries':entries}
new=[]
for spec in r9.fresh_cells():
 cell=r9.Cell(*spec);y=np.zeros_like(cell.y_val)
 for ch,tr in zip(cell.all_chains,cell.traces):
  u=np.asarray(tr['corr'],float)
  for t,idx in enumerate(ch):y[idx]=np.max(u[t+1:]-u[t]) if t+1<len(u) else 0.0
 cell.targets()['signed_best']=y
 fits={name:cell.pipeline(target=name) for name in ['fbg','signed_best']}
 sn={k:(v['em'],v['calls']) for k,v in cell.simple_points().items()}
 row={'tag':cell.tag,'n_train':len(cell.fit_chains),'n_test':cell.n,'signed_target_min':float(y.min()),'signed_target_max':float(y.max()),'negative_training_prefixes':int((y<0).sum()),'targets':{}}
 for name,fit in fits.items():
  row['targets'][name]={'threshold':fit['threshold']}
  for rule in ['guarded','value']:
   p=cell.point(fit['stops'][rule]);v,dom=r9.verdict_eps(p['em'],p['calls'],sn,f'fixed_{cell.bstar}',0.0)
   row['targets'][name][rule]={'em':p['em'],'calls':p['calls'],'category':{'W':'I','T':'F','L':'D'}[v],'dominated_by':dom}
 candidates=[]
 for th in [0,1,2,3]:
  em=[];calls=[]
  for ch in cell.fit_chains:
   chosen=next((i for i in ch if cell.X[i,0]>=1 and cell.X[i,3]<=th),ch[-1]);em.append(1-cell.y_err[chosen]);calls.append(1+2*cell.X[chosen,0])
  candidates.append({'threshold':th,'train_em':100*float(np.mean(em)),'train_calls':float(np.mean(calls))})
 full=100*np.mean([1-cell.y_err[ch[-1]] for ch in cell.fit_chains]);admiss=[c for c in candidates if c['train_em']>=full-1.0]
 selected=min(admiss,key=lambda c:(c['train_calls'],-c['train_em'],c['threshold'])) if admiss else min(candidates,key=lambda c:(-c['train_em'],c['train_calls'],c['threshold']))
 stops=[next((t for t in range(1,cell.width) if f[t,3]<=selected['threshold']),cell.last) for f in cell.feats]
 p=cell.point(stops);extended={**sn,'evidence_growth':(p['em'],p['calls'])}
 row['evidence_growth']={**selected,'em':p['em'],'calls':p['calls'],'training_grid':candidates};row['extended_reference']={}
 for name,st in [('value',cell.v_stop),('guarded',cell.g_stop)]:
  z=cell.point(st);v,dom=r9.verdict_eps(z['em'],z['calls'],extended,f'fixed_{cell.bstar}',0.0);row['extended_reference'][name]={'category':{'W':'I','T':'F','L':'D'}[v],'dominated_by':dom}
 new.append(row);print(cell.tag,row['targets']['signed_best']['guarded'],flush=True)
out['retrospective_analyses']=new
out['summary_new']={name:{rule:{'counts':dict(collections.Counter(c['targets'][name][rule]['category'] for c in new)),'mean_em':float(np.mean([c['targets'][name][rule]['em'] for c in new])),'mean_calls':float(np.mean([c['targets'][name][rule]['calls'] for c in new]))} for rule in ['guarded','value']} for name in ['fbg','signed_best']}
out['summary_extended_reference']={p:dict(collections.Counter(c['extended_reference'][p]['category'] for c in new)) for p in ['value','guarded']}
(ROOT/'results/retrospective/retrospective_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary_new'],indent=2));print('extended',out['summary_extended_reference'])
