"""Analyze every predeclared LS13 horizon and controller; no endpoint calls."""
from pathlib import Path
import sys,json,pickle,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import round9_analyses as r9
from voi_controller import guarded_stop_index
OUT=ROOT/'results/long_horizon';DER=ROOT/'outputs/long_horizon_derived';DER.mkdir(parents=True,exist_ok=True)
r9.MEAS=DER

def main():
 protocol=json.loads((OUT/'protocol.json').read_text()); rows={}
 for split,n in [('train',120),('test',200)]:
  paths=[OUT/'trajectories'/f'{split}-{i:04d}.json' for i in range(n)]
  assert all(p.exists() for p in paths),'Collection incomplete'
  rows[split]=[json.loads(p.read_text()) for p in paths]
  assert [r['id'] for r in rows[split]]==protocol['ids'][split]
 assert not set(protocol['ids']['train']) & set(protocol['ids']['test'])
 results=[];diffs=[];ref=[]
 for H in [4,8]:
  tag=f'ls13_haiku45_h{H}';X=[];yv=[];ye=[];traces=[]
  for r in rows['train']:
   corr=r['prefix_f1'][:H+1]
   for t in range(H+1):
    X.append(r['prefix_features'][t]);ye.append(1-r['prefix_em'][t]);yv.append(max(0,max(corr[t+1:])-corr[t]) if t<H else 0)
   traces.append({'id':r['id'],'corr':corr,'trajectory':r['trajectory'][:H+1]})
  cache={'X_err':X,'X_val':X,'y_val':yv,'y_err':ye,'traces':traces,'feature_names':r9.FEATURES}
  run={'config':{'max_steps':H,'certificate_calibration_fraction':.25,'model':protocol['model']},'lam_main':0,'per_q':[]}
  for r in rows['test']:
   z={**r,'trajectory':r['trajectory'][:H+1],'prefix_em':r['prefix_em'][:H+1],'prefix_features':r['prefix_features'][:H+1]}
   z['answer_repeat_searches']=next((t for t in range(1,H+1) if r9.rx._norm(r['trajectory'][t]['answer'])==r9.rx._norm(r['trajectory'][t-1]['answer'])),H)
   z['primary']={'searches':H};z['frozen_value']={'searches':H};run['per_q'].append(z)
  _,_,chains,fitids=r9.chain_split(run,cache)
  trainem=np.array([r['prefix_em'][:H+1] for r in rows['train']])[fitids]
  ems=100*trainem.mean(0);bstar=next(k for k in range(H+1) if ems[k]>=ems[-1]-1.)
  (DER/f'critic_data_{tag}.pkl').write_bytes(pickle.dumps(cache));(DER/f'run_{tag}.json').write_text(json.dumps(run))
  (DER/f'conf_{tag}.json').write_text(json.dumps({'training_selected_fixed_budget':bstar}))
  c=r9.Cell(tag,f'conf_{tag}.json',f'Fresh Haiku H{H}',tag)
  fit=c.pipeline(rule='guarded') # same threshold for G and V, matching the submitted protocol
  simple=c.simple_points();candidates=[]
  for th in [0,1,2,3]:
   st=[next((t for t in range(1,H+1) if rows['train'][i]['prefix_features'][t][3]<=th),H) for i in fitids]
   candidates.append({'threshold':th,'em':100*float(np.mean([rows['train'][i]['prefix_em'][t] for i,t in zip(fitids,st)])),'calls':float(np.mean(1+2*np.array(st)))})
  ok=[x for x in candidates if x['em']>=ems[-1]-1.]
  sel=min(ok,key=lambda z:(z['calls'],-z['em'],z['threshold'])) if ok else min(candidates,key=lambda z:(-z['em'],z['calls'],z['threshold']))
  egst=[next((t for t in range(1,H+1) if f[t,3]<=sel['threshold']),H) for f in c.feats]
  simple['evidence_growth']=c.point(egst)
  sn={k:(v['em'],v['calls']) for k,v in simple.items()}
  res={'horizon':H,'n_train':120,'n_policy_training':len(fitids),'n_held_out':30,'n_test':200,'threshold':fit['threshold'],'fixed_budget_selected':bstar,'evidence_growth_selection':sel,'simple':{k:{a:v[a] for a in ['em','calls','tokens','latency']} for k,v in simple.items()},'policies':{}}
  for pol in ['guarded','value']:
   p=c.point(fit['stops'][pol]);cat,dom=r9.verdict_eps(p['em'],p['calls'],sn,f'fixed_{bstar}',0.)
   out={a:p[a] for a in ['em','calls','tokens','latency']};out.update(category={'W':'I','T':'F','L':'D'}[cat],dominated_by=dom,contrasts={})
   for axis in ['calls','tokens','latency']:
    pts={k:{**v,'cost':v[axis]} for k,v in simple.items()};h=r9.upper_hull(pts);mix,n1,n2,w=r9.hull_mixture(h,p[axis],pts);d=p['em_vec']-mix
    out['contrasts'][axis]={'mean_delta_em':100*float(d.mean()),'paired_ci':r9.paired_ci(d),'mixture':[n1,n2],'weight_first':w};diffs.append(d);ref.append(out['contrasts'][axis])
   out['versus_selected_fixed']=r9.paired_ci(p['em_vec']-simple[f'fixed_{bstar}']['em_vec']);res['policies'][pol]=out
  full=c.em[:,-1];oracle=c.em.max(1);res['oracle_em']=100*float(oracle.mean());res['oracle_minus_full_ci']=r9.paired_ci(oracle-full)
  results.append(res)
 ds=np.array(diffs).T;influence=(ds-ds.mean(0))/np.sqrt(200*199);se=np.sqrt((influence**2).sum(0));valid=se>0
 rng=np.random.default_rng(212613);z=rng.standard_normal((10000,200))@influence;crit=float(np.quantile(np.max(abs(z[:,valid]/se[valid]),axis=1),.95))
 for j,out in enumerate(ref):
  m=out['mean_delta_em'];out['conditional_simultaneous_ci95']=[m-100*crit*se[j],m+100*crit*se[j]]
  out['conditional_verdict']='win' if out['conditional_simultaneous_ci95'][0]>0 else 'loss' if out['conditional_simultaneous_ci95'][1]<0 else 'unresolved'
 out={'protocol':protocol,'results':results,'family':{'contrasts':12,'critical_value':crit,'method':'10,000 shared-question Gaussian multiplier draws, conditional on observed hull and fitted policies; exploratory, not selection-adjusted','counts':dict(collections.Counter(x['conditional_verdict'] for x in ref))},'horizon8_minus_horizon4_full':r9.paired_ci(np.array([r['prefix_em'][8]-r['prefix_em'][4] for r in rows['test']])),'spend':json.loads((OUT/'spend.json').read_text())}
 (OUT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=[]
 for r in results:
  for pol,label in [('guarded','Guarded'),('value','FBG-only')]:
   p=r['policies'][pol];ci=p['contrasts']['calls']['conditional_simultaneous_ci95']
   lines.append(f"{r['horizon']} & {label} & {r['fixed_budget_selected']} & {p['em']:.2f} & {p['calls']:.2f} & {p['category']} & ${p['contrasts']['calls']['mean_delta_em']:+.2f}$ & $[{ci[0]:+.2f},{ci[1]:+.2f}]$ \\\\")
 (ROOT/'outputs/tables/long_horizon_rows.tex').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'results':results,'family':out['family'],'full8_minus_full4':out['horizon8_minus_horizon4_full'],'spend':out['spend']},indent=2))
if __name__=='__main__':main()
