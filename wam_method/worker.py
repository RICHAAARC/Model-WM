"""One paired training run, all CAL sealing, then one TEST. No retry path."""
import copy
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from wam_qualification.engine import quality
from .io import save,append,Ledger
from .model import Model,train_work,leaf_tail,code_tensor,jpeg_rgb
from .data import image_for
from .protocol import read,messages,CONDITIONS
from .science import scores,aggregate,calibrate,evaluate,decide

def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF);return r.ru_utime+r.ru_stime

class Progress:
    def __init__(self,root,archive):
        self.root=Path(root);self.archive=Path(archive);self.state={'phase':'LOAD','updates':{'B':0,'C':0},
            'train_seconds':{'B':0.,'C':0.},'train_cpu_seconds':{'B':0.,'C':0.},'evaluation_bundles':0}
    def write(self):save(self.root/'progress.json',self.state)
    def seal(self,name):
        # Only closed artifacts are published to Drive. IPC remains local.
        src=self.root/name;dst=self.archive/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():raise RuntimeError('Cannot replace sealed artifact: '+str(dst))
        if src.is_dir():shutil.copytree(src,dst)
        else:shutil.copyfile(src,dst)

def train(model,m,images,root,progress,ledger):
    cfg=m['training'];torch.manual_seed(m['master_seed']);torch.set_num_threads(1)
    training_cpu_start=cpu()
    initial={k:v.detach().cpu().clone() for k,v in model.wam.embedder.state_dict().items()}
    writers={a:copy.deepcopy(model.wam.embedder).to(model.device).eval() for a in 'BC'}
    optimizers={a:torch.optim.AdamW(writers[a].parameters(),lr=cfg['learning_rate'],
        betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay']) for a in 'BC'}
    for writer in writers.values():writer.requires_grad_(True)
    longest={'B':0.,'C':0.};longest_cpu={'B':0.,'C':0.}
    progress.state['phase']='TRAIN';progress.write()
    for step,b in enumerate(m['train']):
        # Alternate order to reduce wall-clock drift bias; both arms get the same bundle.
        for arm in ('BC' if step%2==0 else 'CB'):
            started=time.monotonic();started_cpu=cpu();model.wam.embedder=writers[arm]
            progress.state.update(active_arm=arm,active_started=started);progress.write()
            opt=optimizers[arm];opt.zero_grad(set_to_none=True)
            prefix=f'{arm}/update{step:04d}'
            def update():
                zs=[]
                for t,ident in enumerate(b['images']):
                    with torch.no_grad():
                        z,_,_=train_work(model,image_for(ident,images),messages(b,t),step%2==1,
                            ledger,prefix+f'/probe/{t}')
                    zs.append(z)
                leaf=torch.cat(zs).detach().requires_grad_(True)
                tail=leaf_tail(leaf,code_tensor(b),b['registered_pair'])
                grad=torch.autograd.grad(tail,leaf)[0].detach()
                losses=[]
                for t,ident in enumerate(b['images']):
                    z,base,parts=train_work(model,image_for(ident,images),messages(b,t),step%2==1,
                        ledger,prefix+f'/gradient/{t}')
                    # Recompute exact same forward. Both arms pay both passes;
                    # leaf VJP permits exact T4 rank gradient without four graphs in RAM.
                    if not torch.allclose(z.detach(),zs[t],atol=1e-5,rtol=1e-5):
                        raise RuntimeError('Two-pass gradient forward mismatch')
                    objective=base/4+(z*grad[t*32:(t+1)*32]).sum()*(cfg['tail_lambda'] if arm=='C' else 0.)
                    if not torch.isfinite(objective):raise RuntimeError('Nonfinite training objective')
                    objective.backward();losses.append(parts)
                norm=torch.nn.utils.clip_grad_norm_(writers[arm].parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
                if any(p.grad is not None for p in model.wam.detector.parameters()):
                    raise RuntimeError('Frozen detector received parameter gradients')
                opt.step();torch.cuda.synchronize() if model.device.type=='cuda' else None
                return dict(arm=arm,step=step,tail=float(tail.detach()),grad_norm=float(norm),losses=losses)
            row=ledger.call(prefix,'train_update_'+arm,update)
            elapsed=time.monotonic()-started;used_cpu=cpu()-started_cpu
            longest[arm]=max(longest[arm],elapsed);longest_cpu[arm]=max(longest_cpu[arm],used_cpu)
            progress.state['updates'][arm]+=1;progress.state['train_seconds'][arm]+=elapsed
            progress.state['train_cpu_seconds'][arm]+=used_cpu
            progress.state.update(active_arm=None,active_started=None);progress.write()
            row.update(wall_seconds=elapsed,cpu_seconds=used_cpu)
            name=f'train/{arm}/{step:04d}.json';save(root/name,row);progress.seal(name);progress.write()
            if progress.state['train_seconds'][arm]>m['budgets']['train_arm_seconds']-60:
                raise RuntimeError('Training arm budget exhausted; no shorter substitute')
        if step==15:
            remaining=cfg['updates_per_arm']-16
            prediction={a:remaining*longest[a]*1.25 for a in 'BC'}
            save(root/'training_projection.json',dict(remaining_seconds=prediction,longest=longest,
                remaining_cpu_seconds=sum(longest_cpu.values())*remaining*1.25,
                first_updates_included=16,additional_warmup=0))
            progress.seal('training_projection.json')
            if any(prediction[a]>m['budgets']['train_arm_seconds']-60-progress.state['train_seconds'][a] for a in 'BC'):
                raise RuntimeError('Training completion projection exceeds arm budget')
            if sum(longest_cpu.values())*remaining*1.25+cpu()-training_cpu_start>=m['budgets']['cpu_seconds']-60:
                raise RuntimeError('Training completion projection exceeds total CPU budget')
        if step%16==15:
            name=f'operation_segments/through-update-{step:04d}.jsonl'
            (root/name).parent.mkdir(exist_ok=True)
            shutil.copyfile(root/'operations.jsonl',root/name);progress.seal(name)
    for a in 'BC':
        name=f'checkpoints/{a}-final.pt';(root/name).parent.mkdir(exist_ok=True)
        torch.save({k:v.detach().cpu() for k,v in writers[a].state_dict().items()},root/name)
        progress.seal(name)
    model.wam.embedder=writers['B']
    return initial

def eval_bundle(model,b,arm,images,root,ledger):
    target=root/'evaluation'/arm/b['id'];target.mkdir(parents=True,exist_ok=False)
    rows=[];quality_rows=[]
    for t,ident in enumerate(b['images']):
        clean=image_for(ident,images);media={'null_png':clean,'null_jpeg85':jpeg_rgb(clean)}
        for source in ['registered','foreign']:
            for i,bits in enumerate(messages(b,t,source)):
                role=f'{source}_{i}';key=f'{arm}/{b["id"]}/{t}/write/{role}'
                rgb=ledger.call(key,'evaluation_write',lambda bits=bits:model.eval_write(clean,bits))
                media[role+'_png']=rgb
            mixed=np.rint((media[source+'_0_png'].astype(float)+media[source+'_1_png'].astype(float))/2).astype('uint8')
            media[source+'_mix_png']=mixed;media[source+'_mix_jpeg85']=jpeg_rgb(mixed)
        for c in CONDITIONS:
            rgb=media[c];suffix='.jpg' if 'jpeg85' in c else '.png'
            path=target/f'{t}_{c}{suffix}'
            # Already compressed JPEG must not be compressed twice: save decoded
            # evidence losslessly with .png, while exact first JPEG bytes are recreated below.
            if suffix=='.jpg':
                parent=media[c.replace('jpeg85','png')]
                Image.fromarray(parent).save(path,quality=85,subsampling=2,optimize=False)
            else:Image.fromarray(rgb).save(path)
            with Image.open(path) as im:
                saved=np.asarray(im.convert('RGB')).copy()
            if not np.array_equal(saved,rgb):raise RuntimeError('Saved-image channel mismatch')
            pool,native=ledger.call(f'{arm}/{b["id"]}/{t}/extract/{c}','evaluation_extract',lambda:model.eval_extract(saved))
            np.save(target/f'{t}_{c}.npy',pool,allow_pickle=False)
            row=dict(arm=arm,bundle=b['id'],position=t,image_id=ident,condition=c,
                scores=scores(pool,b['registered'],t),native=native,quality=quality(clean,saved))
            if '_mix_' in c:
                source=c.split('_')[0];truth=messages(b,t,source)
                common=[i for i in range(32) if truth[0][i]==truth[1][i]]
                hard=(np.asarray(pool)[1:].mean((1,2))>0).astype(int)
                row['common_bit_diagnostic']=dict(denominator=len(common),
                    raw_sign_errors=sum(int(hard[i]!=truth[0][i]) for i in common),
                    native_errors=(sum(int(native['bits'][i]!=truth[0][i]) for i in common)
                        if native['status']=='VALID' else None))
            rows.append(row)
            if c in ['registered_0_png','registered_1_png','foreign_0_png','foreign_1_png']:
                source,member,_=c.split('_');truth=messages(b,t,source)[int(member)]
                quality_rows.append(dict(source=source,correct=sum(x==y for x,y in zip(native['bits'] or [],truth)),
                    valid=native['status']=='VALID',**row['quality']))
        # Seven PNG views at native resolution, no output-selected examples.
        names=['null_png','registered_0_png','registered_1_png','registered_mix_png','foreign_0_png','foreign_1_png','foreign_mix_png']
        panel=Image.new('RGB',(512*len(names),512))
        for j,c in enumerate(names):panel.paste(Image.fromarray(media[c]),(512*j,0))
        panel.save(target/f'{t}_visual.png')
    save(target/'readouts.json',rows);summary=aggregate(rows,b);save(target/'bundle.json',summary)
    save(target/'quality.json',quality_rows)
    return summary,quality_rows

def summarize_quality(rows):
    out={}
    for source in ['registered','foreign']:
        xs=[r for r in rows if r['source']==source]
        finite_psnr=[x['psnr_db'] for x in xs if not x.get('psnr_infinite',False)]
        out[source]=dict(denominator=len(xs),valid=sum(x['valid'] for x in xs),
            bit_accuracy=sum(x['correct'] for x in xs)/(32*len(xs)),
            mean_psnr_db=float(np.mean(finite_psnr)) if len(finite_psnr)==len(xs) else None,
            mean_psnr_infinite=len(finite_psnr)!=len(xs),
            min_psnr_db=min(finite_psnr) if finite_psnr else None,
            min_psnr_infinite=not finite_psnr,mean_ssim=float(np.mean([x['ssim'] for x in xs])))
    return out

def main(root,archive,scratch):
    root,archive,scratch=map(Path,(root,archive,scratch));m=read();torch.set_num_threads(1)
    progress=Progress(root,archive);progress.write();ledger=Ledger(root)
    if not torch.cuda.is_available():raise RuntimeError('GPU run must be started by user in Colab')
    paths=json.loads((root/'model_paths.json').read_text())
    model=Model(paths['source'],paths['checkpoint'],'cuda')
    initial=train(model,m,scratch/'images',root,progress,ledger)
    thresholds={};test_results={};qualities={};evaluation_start=time.monotonic()
    progress.state['evaluation_started']=evaluation_start
    raw_test={}
    for split in ['CAL','TEST']:
        if split=='TEST':
            sealed=json.loads((archive/'calibration.json').read_text())
            if set(sealed)!=set('ABC') or not all(x['sealed'] for x in sealed.values()):
                raise RuntimeError('All CAL thresholds must be sealed before TEST')
            thresholds=sealed
        progress.state['phase']=split;progress.write()
        for arm in 'ABC':
            state=initial if arm=='A' else torch.load(root/f'checkpoints/{arm}-final.pt',map_location='cpu',weights_only=True)
            model.wam.embedder.load_state_dict(state,strict=True);model.wam.embedder.requires_grad_(False).eval()
            groups=[];qs=[]
            for b in m['evaluation'][split]:
                result,q=eval_bundle(model,b,arm,scratch/'images',root,ledger)
                groups.append(result);qs.extend(q)
                progress.seal(f'evaluation/{arm}/{b["id"]}')
                progress.state['evaluation_bundles']+=1;progress.write()
                if time.monotonic()-evaluation_start>m['budgets']['evaluation_seconds']-60:
                    raise RuntimeError('Evaluation budget exhausted')
            if split=='CAL':thresholds[arm]=calibrate(groups)
            else:
                raw_test[arm]=groups
                test_results[arm]=[evaluate(b,thresholds[arm]['threshold']) for b in groups]
                qualities[arm]=summarize_quality(qs)
        if split=='CAL':save(root/'calibration.json',thresholds);progress.seal('calibration.json')
    result=decide(test_results,qualities,progress.state['train_seconds'],m['gates'])
    result.update(test=test_results,quality=qualities,operations=ledger.counts,
        training_times=progress.state['train_seconds'])
    paired=[dict(id=test_results['A'][i]['id'],
                 **{a:test_results[a][i] for a in 'ABC'}) for i in range(len(test_results['A']))]
    save(root/'paired_results.json',paired);progress.seal('paired_results.json')
    # Threshold grid is determined only by CAL. Curves are diagnostics and cannot
    # replace the frozen maximum-CAL gate or select a TEST operating point.
    curves={}
    for arm in 'ABC':
        curves[arm]=[]
        for q in [0.,.5,.75,.9,.95,1.]:
            threshold=float(np.quantile(thresholds[arm]['bundle_maxima'],q))
            ev=[evaluate(b,threshold) for b in raw_test[arm]]
            curves[arm].append(dict(cal_quantile=q,threshold=threshold,denominator=len(ev),
                catch_joint=sum(x['catch_all_media'] for x in ev),
                false_attribution=sum(x['false_attribution'] for x in ev),decision_use=False))
    save(root/'risk_curves_diagnostic.json',curves);progress.seal('risk_curves_diagnostic.json')
    expected={k.replace('train_updates_','train_update_'):v for k,v in m['physical_limits'].items() if k!='generate'}
    if ledger.counts!=expected:
        raise RuntimeError('Physical operation denominators disagree')
    save(root/'method_result.json',result);progress.seal('method_result.json')
    progress.state['phase']='COMPLETE';progress.write()

if __name__=='__main__':
    os.environ['MPLBACKEND']='Agg'
    main(*sys.argv[1:4])
