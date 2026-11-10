# Alexey Zakharov (drzoidberg), ROGII team solution.
"""PF/Q deployment and GR calibration."""
import os,pickle,numpy as np,pandas as pd,torch
from scipy.spatial import cKDTree
from phase4_pf import _smoothed_dydx
from phase4_pf_cuda_batch import run_pf_wells_cuda_batch as cudab
NS_F=32; NS_B=16; NP=500; GW=0.30; NUDGE_L=600; VN=PN=0.002; NANCH=30
KC=[0.003,0.03,0.014230249470757707]; KM=[0.001,0.005,0.002]; REG_A,REG_B=5.0,1e-3
BATCH_F=128; BATCH_B=256; ANCH=[(0.25,"25"),(0.5,"50"),(0.75,"75"),(1.0,"100")]

def load_ctx(code_dir, art_dir):
    ks=pickle.load(open(os.path.join(art_dir,"krige_alltrain.pkl"),"rb"))
    try:
        import cupy as cp
        ks._cx_t=cp.asarray(ks.cx.astype(np.float32)); ks._cy_t=cp.asarray(ks.cy.astype(np.float32)); ks._resid_t=cp.asarray(ks.residuals.astype(np.float32)); ks.device="cuda"
    except Exception: ks.device="cpu"
    knots=np.load(os.path.join(art_dir,"sigma_cdf_gkf.npy")); qg=np.linspace(0,1,len(knots))
    seg=pickle.load(open(os.path.join(art_dir,"conf_segments.pkl"),"rb"))
    seg["tree"]=cKDTree(seg["SC"])
    return dict(ks=ks,knots=knots,qg=qg,seg=seg)

def make_kappa(ctx,kv,lo,mid,hi):
    sig=np.sqrt(np.clip(kv,0,None)); q=np.clip(np.interp(sig,ctx["knots"],ctx["qg"]),0,1); return np.interp(q,[0,.5,1.],[hi,mid,lo])

def nudge(be,hw,md):
    pref=hw["TVT_input"].notna().to_numpy()
    if pref.sum()<10: return be
    hi=int(np.where(pref)[0][-1]); dh=_smoothed_dydx(be,md,win_frac=GW)[hi]; pm=md[pref]; pv=(hw["TVT_input"].to_numpy(float)+hw["Z"].to_numpy(float))[pref]
    pdip=float(np.median(_smoothed_dydx(pv,pm,win_frac=GW)[-12:])); s=np.maximum(md-md[hi],0.0); return be+(pdip-dh)*s*np.exp(-s/NUDGE_L)

def confkappa(ctx,hw):
    seg=ctx["seg"]; SC=seg["SC"];SH=seg["SH"];SA=seg["SA"];R=seg["R"];BW=seg["BW"];PG=seg["PGATE"];CS=seg["CS"];SEG=seg["SEG"];KLO=seg["KLO"];KHI=seg["KHI"]
    md=hw.MD.to_numpy(float); X=hw.X.to_numpy(float); Y=hw.Y.to_numpy(float); n=len(md); ws=np.full(n,np.nan); sid=np.floor((md-md[0])/SEG).astype(int)
    for sg in np.unique(sid):
        mm=np.where(sid==sg)[0]; hh=np.hypot(X[mm][-1]-X[mm][0],Y[mm][-1]-Y[mm][0])
        if hh<50: continue
        hx=(X[mm][-1]-X[mm][0])/hh; hy=(Y[mm][-1]-Y[mm][0])/hh
        ii=seg["tree"].query_ball_point((X[mm].mean(),Y[mm].mean()),R)
        if len(ii)<3: continue
        gi=np.array(ii); dot=SH[gi,0]*hx+SH[gi,1]*hy; par=np.abs(dot)>PG; gi=gi[par]; dot=dot[par]
        if len(gi)<3: continue
        dd=np.hypot(SC[gi,0]-X[mm].mean(),SC[gi,1]-Y[mm].mean()); wt=np.exp(-(dd/BW)**2)
        vals=SA[gi]*np.sign(dot); mn=np.sum(wt*vals)/np.sum(wt); ws[mm]=np.sqrt(np.sum(wt*(vals-mn)**2)/np.sum(wt))
    conf=np.where(np.isnan(ws),0.0,np.exp(-ws/CS)); return KLO+conf*(KHI-KLO)

def fit_ab(p_gr,tw_at):
    lo,hi=np.percentile(p_gr,[1,99]); pg=np.clip(p_gr,lo,hi); x=tw_at
    Sxx=x@x+REG_A; Sx=x.sum(); S1=len(x)+REG_B; Sxy=x@pg+REG_A; Sy=pg.sum(); det=Sxx*S1-Sx*Sx
    if abs(det)<1e-9: return 1.0,0.0
    a=(Sxy*S1-Sx*Sy)/det; b=(Sxx*Sy-Sx*Sxy)/det; return float(np.clip(a,0.3,3.0)),float(np.clip(b,-100,100))

def prep_wells(ctx,test_dir,wells):
    """Prepare surface, dip, kappa and typewell inputs."""
    WD={}
    for wid in wells:
        fp=os.path.join(test_dir,wid+"__horizontal_well.csv")
        hw=pd.read_csv(fp).sort_values("MD").reset_index(drop=True)
        md=hw.MD.to_numpy(float); Z=hw.Z.to_numpy(float)
        be,kv=ctx["ks"].predict_batch(hw.X.to_numpy(),hw.Y.to_numpy())
        tw=pd.read_csv(os.path.join(test_dir,wid+"__typewell.csv")).sort_values("TVT"); tt=tw.TVT.values.astype(float); tg=tw.GR.fillna(tw.GR.mean()).values.astype(float)
        pref=hw["TVT_input"].notna().to_numpy()&hw["GR"].notna().to_numpy(); a,b=1.0,0.0
        if pref.sum()>=100:
            p_tvt=hw["TVT_input"].to_numpy(float)[pref]; p_gr=hw["GR"].to_numpy(float)[pref]; a,b=fit_ab(p_gr,np.interp(p_tvt,tt,tg))
        nb=nudge(be,hw,md)
        WD[wid]=dict(hw=hw,md=md,Z=Z,be=be,nb=nb,rkn=_smoothed_dydx(nb,md,win_frac=GW),rkp=_smoothed_dydx(be,md,win_frac=GW),
                     kc=confkappa(ctx,hw),kk=make_kappa(ctx,kv,KC[0],KC[2],KC[1]),km=make_kappa(ctx,kv,KM[0],KM[2],KM[1]),
                     tt=tt,tg=tg,tgc=a*tg+b,ev=(~hw["TVT_input"].notna().to_numpy()))
    return WD

def _fwd(WD,wells,gs,rk_k,kap_k,tw_k,NS=NS_F):
    FP=[]
    for w in wells:
        d=WD[w]; ev=d["ev"]
        FP.append(dict(well=w,job=dict(well=w,hw=d["hw"],tw_tvt=d["tt"],tw_gr=d[tw_k],rk_full=d[rk_k],kap_v_ev=d[kap_k][ev]),ev=ev,md=d["md"][ev]))
    out={}
    for s in range(0,len(FP),BATCH_F):
        c=FP[s:s+BATCH_F]
        for p,o in zip(c,cudab([x["job"] for x in c],NS,NP,0,gs,FLL=1000,vn=VN,pn=PN,dtype=torch.float32)): out[p["well"]]=o[p["ev"]]
    return FP,out

def _bwd(WD,wells,gs,rk_k,kap_k,tw_k,surf_k,frac,NS=NS_B):
    BP=[]
    for w in wells:
        d=WD[w]; hw=d["hw"]; md=d["md"]; Z=d["Z"]; rk=d[rk_k]; kap=d[kap_k]; base=d[surf_k]
        pref=hw.TVT_input.notna().to_numpy(); hi=int(np.where(pref)[0][-1]); surf=base-Z-(base[hi]-(hw.TVT_input.to_numpy(float)[hi]+Z[hi]))
        ev=d["ev"]; ti=np.where(ev)[0]; kf=int(round(frac*(len(ti)-1)))
        if kf<NANCH+10: continue
        anchor=ti[kf]; cov=ti[:kf+1]; rev=np.arange(anchor,ti[0]-1,-1); revmap={int(r):j for j,r in enumerate(rev)}
        md_rev=np.concatenate([[0.],np.cumsum(np.abs(np.diff(md[rev])))]); tin=np.full(len(rev),np.nan); tin[:NANCH]=surf[rev[:NANCH]]
        hwr=pd.DataFrame({"MD":md_rev,"Z":Z[rev],"GR":hw["GR"].to_numpy()[rev],"TVT_input":tin}); etj=[revmap[int(r)] for r in cov]
        BP.append(dict(well=w,job=dict(well=w,hw=hwr,tw_tvt=d["tt"],tw_gr=d[tw_k],rk_full=-rk[rev],kap_v_ev=(kap[rev])[np.isnan(tin)]),etj=etj,cov_md=md[cov]))
    out={}
    for s in range(0,len(BP),BATCH_B):
        c=BP[s:s+BATCH_B]
        for p,o in zip(c,cudab([x["job"] for x in c],NS,NP,0,gs,FLL=1000,vn=VN,pn=PN,dtype=torch.float32)): out[p["well"]]=(p["cov_md"],o[p["etj"]])
    return out


QMEM=[("g30",30.,"rkn","kc","tg","nb"),("g45",45.,"rkn","kc","tg","nb"),("g60",60.,"rkn","kc","tg","nb"),
      ("g120",120.,"rkn","kc","tg","nb"),("mom",60.,"rkp","km","tg","be"),("adjh",60.,"rkn","kk","tgc","nb")]

def run_qadj(ctx,test_dir,wells,beta=0.05):
    WD=prep_wells(ctx,test_dir,wells); wells=[w for w in wells if w in WD]
    FWD={}; BWD={}
    for name,gs,rk,kap,tw,sf in QMEM:
        FP,out=_fwd(WD,wells,gs,rk,kap,tw); FWD[name]={"FP":FP,"out":out}
        BWD[name]={tag:_bwd(WD,wells,gs,rk,kap,tw,sf,frac) for frac,tag in ANCH}

    res={}
    names=[m[0] for m in QMEM]
    for w in wells:
        emd=WD[w]["md"][WD[w]["ev"]]; k=len(emd)
        if k==0: continue
        F=np.stack([FWD[m]["out"][w] for m in names],0)

        def cb(m):
            fr=np.arange(k)/max(k-1,1); o=np.full(k,np.nan); lo=0.0
            byfr={tag:BWD[m][tag].get(w) for _,tag in ANCH}
            a100=byfr["100"]
            for hi,tag in [(0.25,"25"),(0.5,"50"),(0.75,"75"),(1.01,"100")]:
                ms=(fr>=lo)&(fr<hi)
                src=byfr[tag]
                if src is not None:
                    vals=pd.Series(src[1],index=np.round(src[0],3)); o[ms]=vals.reindex(np.round(emd[ms],3)).to_numpy()
                bad=np.isnan(o)&ms
                if bad.any() and a100 is not None:
                    v100=pd.Series(a100[1],index=np.round(a100[0],3)).reindex(np.round(emd[bad],3)).to_numpy(); o[bad]=v100
                lo=hi
            return o
        B=np.stack([cb(m) for m in names],0)
        Mm=np.array([[np.nanmean(np.abs(F[i]-B[j])) for j in range(len(names))] for i in range(len(names))])
        Mm=np.where(np.isfinite(Mm),Mm,np.nanmax(Mm[np.isfinite(Mm)]) if np.isfinite(Mm).any() else 0.0)
        sc=Mm.min(1); Wt=np.exp(-beta*(sc-sc.min())); Wt/=Wt.sum()
        res[w]=(emd, (Wt[:,None]*F).sum(0))
    return res, WD

def run_adjhauto(ctx,test_dir,wells,WD=None):
    """Calibrate horizontal GR and select the per-well GR scale."""
    if WD is None: WD=prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]

    JOBS=[]
    for w in wells:
        d=WD[w]; hw=d["hw"].copy(); tt=d["tt"]; tg=d["tg"]; md=d["md"]
        GR_f=hw["GR"].interpolate(limit_direction="both").fillna(float(np.mean(tg))).to_numpy(float)
        pref=hw["TVT_input"].notna().to_numpy()&hw["GR"].notna().to_numpy(); a,b=1.0,0.0
        if pref.sum()>=100: a,b=fit_ab(hw["GR"].to_numpy(float)[pref],np.interp(hw["TVT_input"].to_numpy(float)[pref],tt,tg))
        grr=(GR_f-b)/a; hw["GR"]=grr


        km=hw["TVT_input"].notna().to_numpy()
        gs=float(np.clip(np.nanstd(grr[km]-np.interp(hw["TVT_input"].to_numpy(float)[km],tt,tg)),10.,60.)) if km.sum()>=2 else 60.
        JOBS.append((w,hw,d,int(round(gs))))

    out={}
    from collections import defaultdict
    bins=defaultdict(list)
    for j in JOBS: bins[j[3]].append(j)
    for gs,items in bins.items():
        FP=[]
        for w,hw,d,_ in items:
            ev=d["ev"]; FP.append(dict(well=w,job=dict(well=w,hw=hw,tw_tvt=d["tt"],tw_gr=d["tg"],rk_full=d["rkn"],kap_v_ev=d["kk"][ev]),ev=ev,md=d["md"][ev]))
        for s in range(0,len(FP),BATCH_F):
            c=FP[s:s+BATCH_F]
            for p,o in zip(c,cudab([x["job"] for x in c],NS_F,NP,0,float(gs),FLL=1000,vn=VN,pn=PN,dtype=torch.float32)): out[p["well"]]=(p["md"],o[p["ev"]])
    return out


def run_segcorr(ctx,test_dir,wells,WD=None,windows=(40,80,160,240,320),beta=0.05,seg_c=10.0):
    """Segment-correlated forward and backward particle filtering."""
    from phase4_pf_cuda_segcorr import run_pf_wells_cuda_batch as cudab_s
    if WD is None: WD=prep_wells(ctx,test_dir,wells)
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
        bjob=dict(well=w,hw=hwr,tw_tvt=d["tt"],tw_gr=d["tg"],rk_full=-rk[rev],kap_v_ev=(kap[rev])[np.isnan(tin)]); etj=[revmap[int(r)] for r in ti]
        P.append(dict(well=w,fjob=fjob,bjob=bjob,etj=etj,ev=ev,emd=md[ti]))
    def runw(SW,kind):
        bs=max(12,min(64,int(96*80/max(SW,80))))
        o={}
        for s in range(0,len(P),bs):
            c=P[s:s+bs]
            for x,r in zip(c,cudab_s([x[kind] for x in c],NS_B,NP,0,60.,FLL=1000,vn=VN,pn=PN,dtype=torch.float32,emission="segcorr",seg_W=SW,seg_c=seg_c)):
                o[x["well"]]= r[x["ev"]] if kind=="fjob" else r[x["etj"]]
            try: torch.cuda.empty_cache()
            except Exception: pass
        return o
    FW={sw:runw(sw,"fjob") for sw in windows}; BW={sw:runw(sw,"bjob") for sw in windows}
    res={}
    for p in P:
        w=p["well"]; Fw=np.stack([FW[sw][w] for sw in windows],0); Bw=np.stack([BW[sw][w] for sw in windows],0); n=len(windows)
        Mm=np.array([[np.mean(np.abs(Fw[i]-Bw[j])) for j in range(n)] for i in range(n)]); sc=beta*(Mm.min(1)-Mm.min(1).min()); e=np.exp(-sc); Wt=e/e.sum()
        res[w]=(p["emd"],(Wt[:,None]*Fw).sum(0))
    return res

def run_jump(ctx,test_dir,wells,WD=None,pjump=0.01,jscale=12.0,jminint=500.0):
    """Forward particle filtering with rate nudging and rare jumps."""
    from phase4_pf_cuda_jump import run_pf_wells_cuda_batch as cudab_j
    if WD is None: WD=prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]; FP=[]
    for w in wells:
        d=WD[w]; ev=d["ev"]
        FP.append(dict(well=w,job=dict(well=w,hw=d["hw"],tw_tvt=d["tt"],tw_gr=d["tg"],rk_full=d["rkn"],kap_v_ev=d["kk"][ev]),ev=ev,md=d["md"][ev]))
    out={}
    for s in range(0,len(FP),BATCH_F):
        c=FP[s:s+BATCH_F]
        for p,o in zip(c,cudab_j([x["job"] for x in c],NS_F,NP,0,60.,FLL=1000,vn=VN,pn=PN,dtype=torch.float32,pjump=pjump,jscale=jscale,jminint=jminint)): out[p["well"]]=(p["md"],o[p["ev"]])
    return out

def run_momentum(ctx,test_dir,wells,WD=None):
    """Forward particle filtering with momentum."""
    if WD is None: WD=prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]
    FP,out=_fwd(WD,wells,60.,"rkp","km","tg")
    return {w:(WD[w]["md"][WD[w]["ev"]],out[w]) for w in out}

def run_gs_composite(ctx,test_dir,wells,gs_list,kappa="kvar",nudge=False,mode="soft",beta=0.05,weights=None,WD=None):
    """Combine GR-scale particle-filter predictions."""
    if WD is None: WD=prep_wells(ctx,test_dir,wells)
    wells=[w for w in wells if w in WD]
    rk="rkn" if nudge else "rkp"; kap="kk" if kappa=="kvar" else "kc"; surf="nb" if nudge else "be"
    FWD={gs:_fwd(WD,wells,float(gs),rk,kap,"tg")[1] for gs in gs_list}
    if mode=="mean":
        wt=np.ones(len(gs_list)) if weights is None else np.asarray(weights,float)
        res={}
        for w in wells:
            emd=WD[w]["md"][WD[w]["ev"]]
            av=[(i,gs) for i,gs in enumerate(gs_list) if w in FWD[gs]]
            if not av: continue
            ps=np.stack([FWD[gs][w] for _,gs in av],0); ww=np.array([wt[i] for i,_ in av],float); ww=ww/ww.sum()
            res[w]=(emd,(ww[:,None]*ps).sum(0))
        return res,WD
    fracs=[1.0] if mode=="soft" else [0.25,0.5,0.75,1.0]; TAG={0.25:"25",0.5:"50",0.75:"75",1.0:"100"}
    BWD={gs:{TAG[f]:_bwd(WD,wells,float(gs),rk,kap,"tg",surf,f) for f in fracs} for gs in gs_list}
    bands=[(0.25,"25"),(0.5,"50"),(0.75,"75"),(1.01,"100")] if mode=="quarters" else [(1.01,"100")]
    res={}
    for w in wells:
        emd=WD[w]["md"][WD[w]["ev"]]; k=len(emd)
        if k==0 or any(w not in FWD[gs] for gs in gs_list): continue
        F=np.stack([FWD[gs][w] for gs in gs_list],0)
        def cb(gs):
            fr=np.arange(k)/max(k-1,1); o=np.full(k,np.nan); lo=0.0; byf={t:BWD[gs][t].get(w) for t in BWD[gs]}; a100=byf.get("100")
            for hi,t in bands:
                ms=(fr>=lo)&(fr<hi); src=byf.get(t)
                if src is not None:
                    o[ms]=pd.Series(src[1],index=np.round(src[0],3)).reindex(np.round(emd[ms],3)).to_numpy()
                bad=np.isnan(o)&ms
                if bad.any() and a100 is not None:
                    o[bad]=pd.Series(a100[1],index=np.round(a100[0],3)).reindex(np.round(emd[bad],3)).to_numpy()
                lo=hi
            return o
        B=np.stack([cb(gs) for gs in gs_list],0)
        Mm=np.array([[np.nanmean(np.abs(F[i]-B[j])) for j in range(len(gs_list))] for i in range(len(gs_list))])
        Mm=np.where(np.isfinite(Mm),Mm,(np.nanmax(Mm[np.isfinite(Mm)]) if np.isfinite(Mm).any() else 0.0))
        sc=Mm.min(1); Wt=np.exp(-beta*(sc-sc.min())); Wt/=Wt.sum(); res[w]=(emd,(Wt[:,None]*F).sum(0))
    return res,WD
