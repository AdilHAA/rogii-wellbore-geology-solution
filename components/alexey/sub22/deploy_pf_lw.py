# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Likelihood-weighted PF/Q deployment."""
import os,numpy as np,pandas as pd,torch
import deploy_pf as dp
from phase4_pf_cuda_logz import run_pf_wells_cuda_batch_perseed as cudab_ps
NS_F,NS_B,NP,VN,PN,NANCH=dp.NS_F,dp.NS_B,dp.NP,dp.VN,dp.PN,dp.NANCH
BATCH_F,BATCH_B,ANCH=dp.BATCH_F,dp.BATCH_B,dp.ANCH
TAU_MULT=6.0; TAU_MULT_E=3.5; BETA_E=0.1

def _lw_weights(res_list,tau_mult=TAU_MULT):
    lzstds=[float(r["logz"].std()) for r in res_list if r["seed_preds"].shape[1]>0]
    tau=max(tau_mult*float(np.median(lzstds)),1e-3)
    return tau

def _fwd_lw(WD,wells,gs,rk_k,kap_k,tw_k,NS=NS_F,tau_mult=TAU_MULT,return_perseed=False):
    FP=[]
    for w in wells:
        d=WD[w]; ev=d["ev"]
        FP.append(dict(well=w,job=dict(well=w,hw=d["hw"],tw_tvt=d["tt"],tw_gr=d[tw_k],rk_full=d[rk_k],kap_v_ev=d[kap_k][ev]),ev=ev))
    allres=[]
    for s in range(0,len(FP),BATCH_F):
        c=FP[s:s+BATCH_F]
        allres.extend(cudab_ps([x["job"] for x in c],NS,NP,0,gs,FLL=1000,vn=VN,pn=PN,dtype=torch.float32))
    tau=_lw_weights(allres,tau_mult)
    out={}; per={}
    for p,r in zip(FP,allres):
        keep=p["ev"][p["job"]["hw"].TVT_input.isna().to_numpy()]
        sp=r["seed_preds"]; lz=r["logz"]
        e=np.exp((lz-lz.max())/tau); wg=e/e.sum()
        out[p["well"]]=(wg[:,None]*sp).sum(0)
        if return_perseed: per[p["well"]]=(sp,lz)
    return (out,tau,per) if return_perseed else (out,tau)

def _bwd_lw(WD,wells,gs,rk_k,kap_k,tw_k,surf_k,frac,NS=NS_B,tau_mult=TAU_MULT):
    BP=[]
    for w in wells:
        d=WD[w]; hw=d["hw"]; md=d["md"]; Z=d["Z"]; rk=d[rk_k]; kap=d[kap_k]; base=d[surf_k]
        pref=hw.TVT_input.notna().to_numpy(); hi=int(np.where(pref)[0][-1]); surf=base-Z-(base[hi]-(hw.TVT_input.to_numpy(float)[hi]+Z[hi]))
        ev=d["ev"]; ti=np.where(ev)[0]; kf=int(round(frac*(len(ti)-1)))
        if kf<NANCH+10: continue
        anchor=ti[kf]; cov=ti[:kf+1]; rev=np.arange(anchor,ti[0]-1,-1); revmap={int(r):j for j,r in enumerate(rev)}
        md_rev=np.concatenate([[0.],np.cumsum(np.abs(np.diff(md[rev])))]); tin=np.full(len(rev),np.nan); tin[:NANCH]=surf[rev[:NANCH]]
        hwr=pd.DataFrame({"MD":md_rev,"Z":Z[rev],"GR":hw["GR"].to_numpy()[rev],"TVT_input":tin})
        etj=np.array([revmap[int(r)] for r in cov])
        BP.append(dict(well=w,job=dict(well=w,hw=hwr,tw_tvt=d["tt"],tw_gr=d[tw_k],rk_full=-rk[rev],kap_v_ev=(kap[rev])[np.isnan(tin)]),etj=etj,cov_md=md[cov]))
    allres=[]
    for s in range(0,len(BP),BATCH_B):
        c=BP[s:s+BATCH_B]
        allres.extend(cudab_ps([x["job"] for x in c],NS,NP,0,gs,FLL=1000,vn=VN,pn=PN,dtype=torch.float32))
    tau=_lw_weights(allres,tau_mult)
    out={}
    for p,r in zip(BP,allres):
        sp=r["seed_preds"]; lz=r["logz"]
        full=np.tile(r["out"],(sp.shape[0],1)); full[:,r["ev_pos"]]=sp
        e=np.exp((lz-lz.max())/tau); wg=e/e.sum()
        out[p["well"]]=(p["cov_md"],(wg[:,None]*full[:,p["etj"]]).sum(0))
    return out

def run_gs_composite_lw(ctx,test_dir,wells,gs_list,kappa="kvar",nudge=False,mode="soft",beta=0.05,WD=None,use_E=False):
    """Combine likelihood-weighted GR-scale particle-filter predictions."""
    if WD is None: WD=dp.prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]
    rk="rkn" if nudge else "rkp"; kap="kk" if kappa=="kvar" else "kc"; surf="nb" if nudge else "be"
    fracs=[1.0] if mode=="soft" else [0.25,0.5,0.75,1.0]; TAG={0.25:"25",0.5:"50",0.75:"75",1.0:"100"}
    BWD={gs:{TAG[f]:_bwd_lw(WD,wells,float(gs),rk,kap,"tg",surf,f) for f in fracs} for gs in gs_list}
    bands=[(0.25,"25"),(0.5,"50"),(0.75,"75"),(1.01,"100")] if mode=="quarters" else [(1.01,"100")]
    def cbw(gs,w,emd):
        k=len(emd); fr=np.arange(k)/max(k-1,1); o=np.full(k,np.nan); lo=0.0
        byf={t:BWD[gs][t].get(w) for t in BWD[gs]}; a100=byf.get("100")
        for hi,t in bands:
            ms=(fr>=lo)&(fr<hi); src=byf.get(t)
            if src is not None:
                o[ms]=pd.Series(src[1],index=np.round(src[0],3)).reindex(np.round(emd[ms],3)).to_numpy()
            bad=np.isnan(o)&ms
            if bad.any() and a100 is not None:
                o[bad]=pd.Series(a100[1],index=np.round(a100[0],3)).reindex(np.round(emd[bad],3)).to_numpy()
            lo=hi
        return o
    FWD={}
    for gs in gs_list:
        if use_E:
            out,tau,per=_fwd_lw(WD,wells,float(gs),rk,kap,"tg",tau_mult=TAU_MULT_E,return_perseed=True)
            outE={}
            for w in wells:
                if w not in per: continue
                sp,lz=per[w]; emd=WD[w]["md"][WD[w]["ev"]]
                cb=cbw(gs,w,emd); ok=np.isfinite(cb)
                e=np.exp((lz-lz.max())/tau)
                if ok.sum()>=20:
                    fbb=np.mean(np.abs(sp[:,ok]-cb[ok][None,:]),axis=1)
                    e=e*np.exp(-BETA_E*(fbb-fbb.min()))
                wg=e/e.sum(); outE[w]=(wg[:,None]*sp).sum(0)
            FWD[gs]=outE
        else:
            FWD[gs]=_fwd_lw(WD,wells,float(gs),rk,kap,"tg")[0]
    res={}
    for w in wells:
        emd=WD[w]["md"][WD[w]["ev"]]; k=len(emd)
        if k==0 or any(w not in FWD[gs] for gs in gs_list): continue
        F=np.stack([FWD[gs][w] for gs in gs_list],0)
        B=np.stack([cbw(gs,w,emd) for gs in gs_list],0)
        n=len(gs_list)
        Mm=np.array([[np.nanmean(np.abs(F[i]-B[j])) for j in range(n)] for i in range(n)])
        Mm=np.where(np.isfinite(Mm),Mm,(np.nanmax(Mm[np.isfinite(Mm)]) if np.isfinite(Mm).any() else 0.0))
        sc=Mm.min(1); Wt=np.exp(-beta*(sc-sc.min())); Wt/=Wt.sum(); res[w]=(emd,(Wt[:,None]*F).sum(0))
    return res,WD

def run_adjhauto_lw(ctx,test_dir,wells,WD=None,tau_mult=TAU_MULT):
    """Likelihood-weighted GR calibration with per-well GR scales."""
    if WD is None: WD=dp.prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]
    JOBS=[]
    for w in wells:
        d=WD[w]; hw=d["hw"].copy(); tt=d["tt"]; tg=d["tg"]
        GR_f=hw["GR"].interpolate(limit_direction="both").fillna(float(np.mean(tg))).to_numpy(float)
        pref=hw["TVT_input"].notna().to_numpy()&hw["GR"].notna().to_numpy(); a,b=1.0,0.0
        if pref.sum()>=100: a,b=dp.fit_ab(hw["GR"].to_numpy(float)[pref],np.interp(hw["TVT_input"].to_numpy(float)[pref],tt,tg))
        grr=(GR_f-b)/a; hw["GR"]=grr
        km=hw["TVT_input"].notna().to_numpy()
        gs=1.30*float(np.clip(np.nanstd(grr[km]-np.interp(hw["TVT_input"].to_numpy(float)[km],tt,tg)),10.,60.)) if km.sum()>=2 else 60.
        JOBS.append((w,hw,d,int(round(gs))))
    from collections import defaultdict
    bins=defaultdict(list)
    for j in JOBS: bins[j[3]].append(j)
    collected=[]
    for gs,items in bins.items():
        FP=[]
        for w,hw,d,_ in items:
            ev=d["ev"]; FP.append(dict(well=w,job=dict(well=w,hw=hw,tw_tvt=d["tt"],tw_gr=d["tg"],rk_full=d["rkn"],kap_v_ev=d["kk"][ev]),ev=ev,md=d["md"][ev]))
        for s in range(0,len(FP),BATCH_F):
            c=FP[s:s+BATCH_F]
            res=cudab_ps([x["job"] for x in c],NS_F,NP,0,float(gs),FLL=1000,vn=VN,pn=PN,dtype=torch.float32)
            for p,r in zip(c,res): collected.append((p["well"],p["md"],r))
    tau=_lw_weights([r for _,_,r in collected],tau_mult)
    out={}
    for w,md,r in collected:
        sp=r["seed_preds"]; lz=r["logz"]; e=np.exp((lz-lz.max())/tau); wg=e/e.sum()
        out[w]=(md,(wg[:,None]*sp).sum(0))
    return out

def run_qadj_lw(ctx,test_dir,wells,beta=0.10,WD=None,members=("g30","g45","g60","mom","adjh")):
    """Combine likelihood-weighted forward and backward particle-filter predictions."""
    if WD is None: WD=dp.prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]
    SPEC={m:s for m in members for s in [dict(g30=(30.,"rkn","kc","tg","nb"),g45=(45.,"rkn","kc","tg","nb"),
          g60=(60.,"rkn","kc","tg","nb"),g120=(120.,"rkn","kc","tg","nb"),
          mom=(60.,"rkp","km","tg","be"),adjh=(60.,"rkn","kk","tgc","nb"))[m]]}
    FWD={}; BWD={}
    for m in members:
        gs,rk,kap,tw,sf=SPEC[m]
        FWD[m]=_fwd_lw(WD,wells,gs,rk,kap,tw)[0]
        BWD[m]={tag:_bwd_lw(WD,wells,gs,rk,kap,tw,sf,frac) for frac,tag in ANCH}
    res={}
    for w in wells:
        emd=WD[w]["md"][WD[w]["ev"]]; k=len(emd)
        if k==0 or any(w not in FWD[m] for m in members): continue
        F=np.stack([FWD[m][w] for m in members],0)
        def cb(m):
            fr=np.arange(k)/max(k-1,1); o=np.full(k,np.nan); lo=0.0
            byf={tag:BWD[m][tag].get(w) for _,tag in ANCH}; a100=byf.get("100")
            for hi,tag in [(0.25,"25"),(0.5,"50"),(0.75,"75"),(1.01,"100")]:
                ms=(fr>=lo)&(fr<hi); src=byf.get(tag)
                if src is not None:
                    o[ms]=pd.Series(src[1],index=np.round(src[0],3)).reindex(np.round(emd[ms],3)).to_numpy()
                bad=np.isnan(o)&ms
                if bad.any() and a100 is not None:
                    o[bad]=pd.Series(a100[1],index=np.round(a100[0],3)).reindex(np.round(emd[bad],3)).to_numpy()
                lo=hi
            return o
        B=np.stack([cb(m) for m in members],0); n=len(members)
        Mm=np.array([[np.nanmean(np.abs(F[i]-B[j])) for j in range(n)] for i in range(n)])
        Mm=np.where(np.isfinite(Mm),Mm,(np.nanmax(Mm[np.isfinite(Mm)]) if np.isfinite(Mm).any() else 0.0))
        sc=Mm.min(1); Wt=np.exp(-beta*(sc-sc.min())); Wt/=Wt.sum(); res[w]=(emd,(Wt[:,None]*F).sum(0))
    return res,WD

def run_segcorr_lz(ctx,test_dir,wells,WD=None,windows=(40,80,160,240,320),beta=0.05,seg_c=10.0,tau_mult=TAU_MULT):
    """Segment-correlated particle filtering with evidence-weighted seeds."""
    from phase4_pf_cuda_segcorr_logz import run_pf_wells_cuda_batch_perseed as cudab_sps
    if WD is None: WD=dp.prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]
    P=[]
    for w in wells:
        d=WD[w]; hw=d["hw"]; md=d["md"]; Z=d["Z"]; nb=d["nb"]; rk=d["rkn"]; kap=d["kk"]
        pref=hw.TVT_input.notna().to_numpy(); hi=int(np.where(pref)[0][-1]); surf=nb-Z-(nb[hi]-(hw.TVT_input.to_numpy(float)[hi]+Z[hi]))
        ev=d["ev"]; ti=np.where(ev)[0]
        if len(ti)<NANCH+40: continue
        fjob=dict(well=w,hw=hw,tw_tvt=d["tt"],tw_gr=d["tg"],rk_full=rk,kap_v_ev=kap[ev])
        rev=np.arange(len(md)-1,ti[0]-1,-1); revmap={int(r):j for j,r in enumerate(rev)}
        md_rev=np.concatenate([[0.],np.cumsum(np.abs(np.diff(md[rev])))]); tin=np.full(len(rev),np.nan); tin[:NANCH]=surf[rev[:NANCH]]
        hwr=pd.DataFrame({"MD":md_rev,"Z":Z[rev],"GR":hw["GR"].to_numpy()[rev],"TVT_input":tin})
        bjob=dict(well=w,hw=hwr,tw_tvt=d["tt"],tw_gr=d["tg"],rk_full=-rk[rev],kap_v_ev=(kap[rev])[np.isnan(tin)])
        etj=np.array([revmap[int(r)] for r in ti])
        P.append(dict(well=w,fjob=fjob,bjob=bjob,etj=etj,ev=ev,emd=md[ti]))
    def runw(SW,kind):
        bs=max(12,min(64,int(96*80/max(SW,80))))
        allr=[]
        for s in range(0,len(P),bs):
            c=P[s:s+bs]
            allr.extend(cudab_sps([x[kind] for x in c],NS_B,NP,0,60.,FLL=1000,vn=VN,pn=PN,dtype=torch.float32,emission="segcorr",seg_W=SW,seg_c=seg_c))
            try: torch.cuda.empty_cache()
            except Exception: pass
        tau=_lw_weights(allr,tau_mult); o={}
        for x,r in zip(P,allr):
            sp=r["seed_preds"]; lz=r["logz"]; e=np.exp((lz-lz.max())/tau); wg=e/e.sum()
            if kind=="fjob":
                o[x["well"]]=(wg[:,None]*sp[:,x["ev"][x["fjob"]["hw"].TVT_input.isna().to_numpy()]]).sum(0)
            else:
                full=np.tile(r["out"],(sp.shape[0],1)); full[:,r["ev_pos"]]=sp
                o[x["well"]]=(wg[:,None]*full[:,x["etj"]]).sum(0)
        return o
    FW={sw:runw(sw,"fjob") for sw in windows}; BW={sw:runw(sw,"bjob") for sw in windows}
    res={}
    for p in P:
        w=p["well"]
        if any(w not in FW[sw] or w not in BW[sw] for sw in windows): continue
        Fw=np.stack([FW[sw][w] for sw in windows],0); Bw=np.stack([BW[sw][w] for sw in windows],0); n=len(windows)
        Mm=np.array([[np.mean(np.abs(Fw[i]-Bw[j])) for j in range(n)] for i in range(n)])
        sc=beta*(Mm.min(1)-Mm.min(1).min()); e=np.exp(-sc); Wt=e/e.sum()
        res[w]=(p["emd"],(Wt[:,None]*Fw).sum(0))
    return res
