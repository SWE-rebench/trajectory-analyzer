#!/usr/bin/env python3
"""Repeat repository-holdout fingerprint retrieval from canonical parser-v1.1 metrics."""
from __future__ import annotations
import csv,hashlib,json,math,os,random
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Archived source: the original research workspace and frozen inputs are required.
ROOT=Path(os.environ.get('BEHAVIORAL_RESEARCH_WORKSPACE', '.')).resolve()
STAGE=Path(__file__).resolve().parent
R=ROOT/'analysis/evolution_134k/workshop_redesign_v1'
FROZEN=STAGE.parent/'frozen';OUT=STAGE/'repository'
FULL=R/'fingerprint_repository_holdout_full_v1'
CANONICAL=STAGE.parent/'outputs/python_rollout_metrics_parser_v4_v1.parquet'
PRIOR=R/'fingerprint_recalc_parser_v1_1_v1/primary'
LOCK=FROZEN/'PARSER_V4_FINGERPRINT_INPUTS_v1_1.lock.json'
FEATURES=['search_navigation_share','file_read_share','source_write_share','formal_test_share','action_entropy','action_switch_rate','final_diff_inspected','termination_after_write_without_validation']
BLOCKS={'full':FEATURES,'composition':FEATURES[:4],'sequence_and_closure':FEATURES[4:]}
BOOT_DRAWS=500;PERM_DRAWS=10000

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def seed(s):return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)
def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(p,fields,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def percentile(x,q):
 x=sorted(x);pos=(len(x)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
 return x[lo] if lo==hi else x[lo]*(hi-pos)+x[hi]*(pos-lo)
def cosine(a,b):
 dot=sum(x*y for x,y in zip(a,b));na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(y*y for y in b))
 return dot/(na*nb) if na and nb else float('nan')
def close(a,b,tol=1e-12):return abs(a-b)<=tol

def connected(obs,stacks):
 adj=defaultdict(set)
 for tid,sid,_,_ in obs:adj['t:'+tid].add('s:'+sid);adj['s:'+sid].add('t:'+tid)
 start='s:'+stacks[0];seen={start};todo=[start]
 while todo:
  n=todo.pop()
  for q in adj[n]:
   if q not in seen:seen.add(q);todo.append(q)
 return all('s:'+s in seen for s in stacks),len(seen)
def fit(obs,stacks):
 by_t=defaultdict(list);by_s=defaultdict(list)
 for tid,sid,y,repo in obs:by_t[tid].append((sid,y));by_s[sid].append((tid,y,repo))
 beta={s:0.0 for s in stacks};alpha={t:0.0 for t in by_t}
 for iterations in range(1,10001):
  for t,vals in by_t.items():alpha[t]=sum(y-beta[s] for s,y in vals)/len(vals)
  new={s:sum(y-alpha[t] for t,y,_ in vals)/len(vals) for s,vals in by_s.items()}
  center=sum(new.values())/len(stacks)
  for s in new:new[s]-=center
  for t in alpha:alpha[t]+=center
  delta=max(abs(new[s]-beta[s]) for s in stacks);beta=new
  if delta<1e-12:break
 if iterations>=10000:raise RuntimeError('Fixed-effect solver did not converge')
 adjusted=[];ss=0.0
 for t,s,y,repo in obs:
  a=y-alpha[t];res=y-alpha[t]-beta[s];ss+=res*res;adjusted.append((repo,t,s,a))
 return alpha,beta,adjusted,iterations,math.sqrt(ss/len(obs))
def retrieve(profiles,adjusted,stacks,features,scenario,dump=False):
 matrix=[];matches=[];stats=[]
 for qfold,cfold in [('A','B'),('B','A')]:
  scales={f:math.sqrt(sum(v*v for _,_,_,v in adjusted[(qfold,f)])/len(adjusted[(qfold,f)])) for f in features}
  sims={}
  for qs in stacks:
   q=[profiles[(qfold,f,qs)]/scales[f] for f in features]
   for cs in stacks:
    c=[profiles[(cfold,f,cs)]/scales[f] for f in features];val=cosine(q,c);sims[(qs,cs)]=val
    if dump:matrix.append({'scenario':scenario,'query_fold':qfold,'candidate_fold':cfold,'query_stack_id':qs,'candidate_stack_id':cs,'cosine_similarity':val,'features_used':len(features)})
  top1=top3=0;ranks=[];selfs=[];nons=[];nearest=[]
  for qs in stacks:
   ordered=sorted(stacks,key=lambda cs:(-sims[(qs,cs)],cs));rank=ordered.index(qs)+1;ss=sims[(qs,qs)];bn=max(sims[(qs,cs)] for cs in stacks if cs!=qs)
   top1+=rank==1;top3+=rank<=3;ranks.append(rank);selfs.append(ss);nons.append(bn);nearest.append(stacks.index(ordered[0]))
   if dump:matches.append({'scenario':scenario,'query_fold':qfold,'candidate_fold':cfold,'query_stack_id':qs,'retrieved_stack_id':ordered[0],'self_rank':rank,'self_similarity':ss,'best_nonself_similarity':bn,'self_margin':ss-bn,'correct_top1':rank==1,'correct_top3':rank<=3})
  stats.append({'query_fold':qfold,'candidate_fold':cfold,'top1':top1/len(stacks),'top3':top3/len(stacks),'mean_rank':sum(ranks)/len(ranks),'mean_self':sum(selfs)/len(selfs),'mean_nonself':sum(nons)/len(nons),'mean_margin':sum(a-b for a,b in zip(selfs,nons))/len(stacks),'nearest_indices':nearest})
 agg={k:sum(d[k] for d in stats)/2 for k in ['top1','top3','mean_rank','mean_self','mean_nonself','mean_margin']}
 return matrix,matches,stats,agg
def bootstrap_profiles(adjusted,fold_repos,stacks,rng):
 profiles={};sampled={}
 for fold in ('A','B'):
  mult=Counter(rng.choice(fold_repos[fold]) for _ in fold_repos[fold])
  for f in FEATURES:
   vals=defaultdict(list);expanded=[]
   for repo,tid,sid,v in adjusted[(fold,f)]:
    for _ in range(mult.get(repo,0)):vals[sid].append(v);expanded.append((repo,tid,sid,v))
   if any(not vals[s] for s in stacks):return None,None
   raw={s:sum(vals[s])/len(vals[s]) for s in stacks};center=sum(raw.values())/len(stacks)
   for s in stacks:profiles[(fold,f,s)]=raw[s]-center
   sampled[(fold,f)]=expanded
 return profiles,sampled

def main():
 lock=json.loads(LOCK.read_text())
 for item in lock['sources'].values():
  if sha(Path(item['path']))!=item['sha256']:raise RuntimeError(f"Input hash mismatch: {item['path']}")
 full_lock=json.loads((FULL/'frozen/FULL_REGISTRY_FREEZE_v1.lock.json').read_text())
 for name,digest in full_lock['output_sha256'].items():
  if sha(FULL/'frozen'/name)!=digest:raise RuntimeError(f'Inherited registry drift: {name}')
 stack_rows=read_csv(FULL/'frozen/full_stack_registry_v1.csv');roll=read_csv(FULL/'frozen/full_rollout_registry_v1.csv')
 stacks=sorted(r['stack_id'] for r in stack_rows);meta={r['stack_id']:r for r in stack_rows};selected={r['rollout_id'] for r in roll}
 lookup={r['rollout_id']:(r['fold'],r['repository'],r['task_id'],r['stack_id']) for r in roll}
 if len(stacks)!=111 or len(selected)!=68022:raise RuntimeError('Inherited population drift')
 columns=['rollout_id','task_id','stack_id','parser_id','exploration_share','read_share','non_test_repository_write_without_observed_failure_share','formal_test_invocations_per_semantic_action','action_entropy_normalized','action_switch_rate','post_edit_diff_inspection','unverified_termination']
 table=pq.read_table(CANONICAL,columns=columns);table=table.filter(pc.is_in(table['rollout_id'],value_set=pa.array(sorted(selected))))
 canonical={r['rollout_id']:r for r in table.to_pylist()}
 if set(canonical)!=selected:raise RuntimeError(f'Canonical coverage {len(canonical)}/{len(selected)}')
 prior_rows=pq.read_table(PRIOR/'rollout_features_parser_v1_1_v1.parquet').to_pylist();prior={(r['rollout_id'],r['feature_id']):r for r in prior_rows if r['rollout_id'] in selected}
 if len(prior)!=len(selected)*len(FEATURES):raise RuntimeError('Prior feature grid drift')
 rollout_rows=[];changed=Counter();paired_delta=defaultdict(list);definedness=Counter();raw=defaultdict(list)
 for rid in sorted(selected):
  fold,repo,tid,sid=lookup[rid];r=canonical[rid]
  if r['task_id']!=tid or r['stack_id']!=sid or r['parser_id']!='behavioral-parser-4.0.0+git.4ec73bdbd306':raise RuntimeError(f'Canonical identity mismatch {rid}')
  explore=r['exploration_share'];read=r['read_share']
  new_vals={'search_navigation_share':None if explore is None or read is None else float(explore)-float(read),'file_read_share':None if read is None else float(read),'source_write_share':None if r['non_test_repository_write_without_observed_failure_share'] is None else float(r['non_test_repository_write_without_observed_failure_share']),'formal_test_share':None if r['formal_test_invocations_per_semantic_action'] is None else float(r['formal_test_invocations_per_semantic_action']),'action_entropy':None if r['action_entropy_normalized'] is None else float(r['action_entropy_normalized']),'action_switch_rate':None if r['action_switch_rate'] is None else float(r['action_switch_rate']),'final_diff_inspected':float(bool(r['post_edit_diff_inspection'])),'termination_after_write_without_validation':float(r['unverified_termination']=='unverified')}
  for f in FEATURES:
   old=prior[(rid,f)];old_def=bool(old['parser_v1_1_defined']) and old['parser_v1_1_value'] is not None;old_value=float(old['parser_v1_1_value']) if old_def else None;new=new_vals[f];new_def=new is not None;delta=new-old_value if new_def and old_def else None
   if delta is not None:paired_delta[f].append(delta);changed[f]+=not close(delta,0.0)
   definedness[f]+=old_def!=new_def
   rollout_rows.append({'rollout_id':rid,'fold':fold,'repository':repo,'task_id':tid,'stack_id':sid,'feature_id':f,'parser_v1_1_value':old_value,'parser_v1_1_defined':old_def,'parser_v4_value':new,'parser_v4_defined':new_def,'delta':delta,'provenance':'canonical_parser_v4_rollout_registry'})
   if new_def:raw[(fold,repo,tid,sid,f)].append(float(new))
 rec_rows=[]
 for f in FEATURES:
  vals=paired_delta[f];rec_rows.append({'feature_id':f,'selected_rollouts':len(selected),'paired_defined_rollouts':len(vals),'definedness_transitions':definedness[f],'changed_rollouts':changed[f],'mean_paired_delta':sum(vals)/len(vals) if vals else '','max_absolute_delta':max(map(abs,vals)) if vals else ''})
 OUT.mkdir(parents=True,exist_ok=True)
 schema=pa.schema([('rollout_id',pa.string()),('fold',pa.string()),('repository',pa.string()),('task_id',pa.string()),('stack_id',pa.string()),('feature_id',pa.string()),('parser_v1_1_value',pa.float64()),('parser_v1_1_defined',pa.bool_()),('parser_v4_value',pa.float64()),('parser_v4_defined',pa.bool_()),('delta',pa.float64()),('provenance',pa.string())])
 pq.write_table(pa.Table.from_pylist(rollout_rows,schema=schema),OUT/'rollout_features_parser_v4_v1.parquet',compression='zstd')

 obs=defaultdict(list);cell_rows=[]
 for (fold,repo,tid,sid,f),vals in sorted(raw.items()):
  if len(vals)>=2:
   y=sum(vals)/len(vals);obs[(fold,f)].append((tid,sid,y,repo));cell_rows.append({'fold':fold,'repository':repo,'task_id':tid,'stack_id':sid,'feature_id':f,'task_stack_mean':y,'defined_rollouts':len(vals)})
 profiles={};adjusted={};diag=[]
 for fold in ('A','B'):
  for f in FEATURES:
   ok,nodes=connected(obs[(fold,f)],stacks)
   if not ok:raise RuntimeError(f'Disconnected {fold} {f}')
   alpha,beta,adj,it,rmse=fit(obs[(fold,f)],stacks);adjusted[(fold,f)]=adj
   for s,v in beta.items():profiles[(fold,f,s)]=v
   diag.append({'fold':fold,'feature_id':f,'task_stack_cells':len(obs[(fold,f)]),'tasks':len({x[0] for x in obs[(fold,f)]}),'stacks':len({x[1] for x in obs[(fold,f)]}),'connected':ok,'solver_iterations':it,'residual_rmse':rmse})
 profile_rows=[{'fold':fold,'stack_id':s,'feature_id':f,'task_adjusted_coordinate':profiles[(fold,f,s)]} for fold in ('A','B') for s in stacks for f in FEATURES]
 matrix,matches,primary_stats,primary=retrieve(profiles,adjusted,stacks,FEATURES,'all_stacks_full8',True)
 scenario_rows=[{'scenario':'all_stacks_full8','stacks':len(stacks),'features':8,**primary}]
 for block in ('composition','sequence_and_closure'):
  _,_,_,a=retrieve(profiles,adjusted,stacks,BLOCKS[block],f'all_stacks_{block}',False);scenario_rows.append({'scenario':f'all_stacks_{block}','stacks':len(stacks),'features':4,**a})
 swelike=sorted(s for s in stacks if meta[s]['scaffold']=='swelike-py')
 _,_,_,a=retrieve(profiles,adjusted,swelike,FEATURES,'swelike_py_full8',False);scenario_rows.append({'scenario':'swelike_py_full8','stacks':len(swelike),'features':8,**a})

 fold_repos={fold:sorted({r['repository'] for r in roll if r['fold']==fold}) for fold in ('A','B')};rng=random.Random(seed('full-fingerprint-bootstrap-v1'));boot=[]
 for draw in range(BOOT_DRAWS):
  bp,ba=bootstrap_profiles(adjusted,fold_repos,stacks,rng)
  if bp is None:continue
  _,_,_,x=retrieve(bp,ba,stacks,FEATURES,'bootstrap',False);boot.append({'draw':draw,**x})
 if len(boot)<490:raise RuntimeError(f'Only {len(boot)} valid bootstrap draws')
 prng=random.Random(seed('full-fingerprint-permutation-v1'));perm=[];exceed=0
 for draw in range(PERM_DRAWS):
  vals=[]
  for d in primary_stats:
   labels=list(stacks);prng.shuffle(labels);correct=sum(labels[idx]==qs for qs,idx in zip(stacks,d['nearest_indices']));vals.append(correct/len(stacks))
  v=sum(vals)/2;exceed+=v>=primary['top1']-1e-15;perm.append({'draw':draw,'bidirectional_top1':v})
 pvalue=(exceed+1)/(PERM_DRAWS+1);bt=[r['top1'] for r in boot]

 old_summary=json.loads((PRIOR/'repository_holdout_summary_v1.json').read_text());old_scen={'all_stacks_full8':old_summary['primary'],**{r['scenario']:r for r in old_summary['sensitivities']}}
 compare=[]
 for row in scenario_rows:
  old=old_scen[row['scenario']]
  compare.append({'scenario':row['scenario'],'stacks':row['stacks'],'parser_v1_1_top1':old['top1'],'parser_v4_top1':row['top1'],'delta_top1':row['top1']-old['top1'],'parser_v1_1_top3':old['top3'],'parser_v4_top3':row['top3'],'delta_top3':row['top3']-old['top3'],'parser_v1_1_mean_rank':old['mean_rank'],'parser_v4_mean_rank':row['mean_rank'],'delta_mean_rank':row['mean_rank']-old['mean_rank'],'parser_v1_1_mean_margin':old['mean_margin'],'parser_v4_mean_margin':row['mean_margin'],'delta_mean_margin':row['mean_margin']-old['mean_margin']})
 old_matches={(r['query_fold'],r['query_stack_id']):r for r in read_csv(PRIOR/'retrieval_matches_parser_v1_1_v1.csv') if r['scenario']=='all_stacks_full8'}
 transition=[];tc=Counter()
 for r in matches:
  old=old_matches[(r['query_fold'],r['query_stack_id'])];oc=old['correct_top1']=='True';nc=bool(r['correct_top1']);label='both_correct' if oc and nc else 'parser_v1_1_only' if oc else 'parser_v4_only' if nc else 'neither';tc[label]+=1
  transition.append({'query_fold':r['query_fold'],'candidate_fold':r['candidate_fold'],'query_stack_id':r['query_stack_id'],'parser_v1_1_retrieved_stack_id':old['retrieved_stack_id'],'parser_v4_retrieved_stack_id':r['retrieved_stack_id'],'parser_v1_1_self_rank':old['self_rank'],'parser_v4_self_rank':r['self_rank'],'rank_delta':int(r['self_rank'])-int(old['self_rank']),'parser_v1_1_self_margin':old['self_margin'],'parser_v4_self_margin':r['self_margin'],'margin_delta':float(r['self_margin'])-float(old['self_margin']),'correctness_transition':label})
 old_profiles={(r['fold'],r['stack_id'],r['feature_id']):float(r['task_adjusted_coordinate']) for r in read_csv(PRIOR/'stack_profiles_parser_v1_1_v1.csv')}
 profile_comp=[{'fold':r['fold'],'stack_id':r['stack_id'],'feature_id':r['feature_id'],'parser_v1_1_coordinate':old_profiles[(r['fold'],r['stack_id'],r['feature_id'])],'parser_v4_coordinate':r['task_adjusted_coordinate'],'delta':r['task_adjusted_coordinate']-old_profiles[(r['fold'],r['stack_id'],r['feature_id'])]} for r in profile_rows]

 summary={'schema_version':'repository-holdout-fingerprint-parser-v4-summary-v1','parser_version':'behavioral-parser-4.0.0+git.4ec73bdbd306','input_freeze_id':lock['freeze_id'],'population':'evaluated_final_v2','legacy_exact_0_used':False,'outcomes_accessed':False,'human_or_model_panel_labels_used':False,'absolute_length_or_specific_tool_identity_used':False,'stacks':len(stacks),'canonical_models':len({meta[s]['canonical_model'] for s in stacks}),'scaffolds':len({meta[s]['scaffold'] for s in stacks}),'repositories':{'A':len(fold_repos['A']),'B':len(fold_repos['B'])},'selected_rollouts':len(selected),'selected_parse_failures':0,'features':FEATURES,'primary':{**primary,'forward_top1':primary_stats[0]['top1'],'reverse_top1':primary_stats[1]['top1'],'chance_top1':1/len(stacks),'bootstrap_ci95_top1':[percentile(bt,.025),percentile(bt,.975)],'bootstrap_valid_draws':len(boot),'permutation_pvalue_top1':pvalue},'sensitivities':scenario_rows[1:],'comparison_to_parser_v1_1':{'old_top1':old_summary['primary']['top1'],'delta_top1':primary['top1']-old_summary['primary']['top1'],'old_top3':old_summary['primary']['top3'],'delta_top3':primary['top3']-old_summary['primary']['top3'],'query_transitions':dict(tc)},'support':{'canonical_rollout_rows':len(canonical),'rollout_feature_rows':len(rollout_rows),'task_stack_feature_cells':len(cell_rows),'profiles':len(profile_rows),'fits':len(diag)},'input_sha256':{'canonical_rollout_metrics':sha(CANONICAL),'prior_parser_features':sha(PRIOR/'rollout_features_parser_v1_1_v1.parquet'),'input_lock':sha(LOCK)}}
 outputs=[
  ('task_stack_features_parser_v4_v1.csv',cell_rows),('stack_profiles_parser_v4_v1.csv',profile_rows),('fit_diagnostics_parser_v4_v1.csv',diag),('similarity_matrix_parser_v4_v1.csv',matrix),('retrieval_matches_parser_v4_v1.csv',matches),('scenario_summary_parser_v4_v1.csv',scenario_rows),('bootstrap_parser_v4_v1.csv',boot),('permutation_parser_v4_v1.csv',perm),('feature_transition_parser_v4_v1.csv',rec_rows),('scenario_transition_parser_v4_v1.csv',compare),('query_transition_parser_v4_v1.csv',transition),('profile_coordinate_transition_parser_v4_v1.csv',profile_comp)]
 for name,rows in outputs:write_csv(OUT/name,list(rows[0]),rows)
 (OUT/'repository_holdout_summary_v4.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary['primary'],indent=2));print(json.dumps(summary['comparison_to_parser_v1_1'],indent=2));print(json.dumps(rec_rows,indent=2))
if __name__=='__main__':main()
