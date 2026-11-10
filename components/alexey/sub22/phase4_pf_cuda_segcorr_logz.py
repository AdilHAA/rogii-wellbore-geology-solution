# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Segment-correlated CUDA particle filter with per-seed log evidence."""
import numpy as np, torch

def _interp_nan_linear_pos(y, fill):
    y = np.asarray(y, dtype=np.float64).copy(); bad = np.isnan(y)
    if bad.any():
        good = np.flatnonzero(~bad)
        if good.size == 0: y[:] = fill
        else: y[bad] = np.interp(np.arange(y.size)[bad], good, y[good])
    return y

def _prep(hw, rk_full, kap_v_ev, tw_gr_mean):
    tvt_all = hw["TVT_input"].to_numpy(dtype=np.float64, copy=True); z_all = hw["Z"].to_numpy(np.float64); md_all = hw["MD"].to_numpy(np.float64)
    gr_all = _interp_nan_linear_pos(hw["GR"].to_numpy(np.float64), tw_gr_mean)
    known = ~np.isnan(tvt_all); ev_pos = np.flatnonzero(~known); out = tvt_all.copy()
    if ev_pos.size == 0: return {"n":0,"out":out,"ev_pos":ev_pos}
    kn_pos = np.flatnonzero(known); last_pos = kn_pos[-1]
    last_tvt=float(tvt_all[last_pos]); last_Z=float(z_all[last_pos]); last_MD=float(md_all[last_pos])
    tail=kn_pos[-30:]; dt=np.diff(tvt_all[tail]); dz=np.diff(z_all[tail]); dmm=np.diff(md_all[tail]); mm=dmm>0
    ir=float(np.median((dt[mm]+dz[mm])/dmm[mm])) if mm.sum()>=3 else 0.0
    md_ev=md_all[ev_pos]; prev=np.empty_like(md_ev); prev[0]=last_MD; prev[1:]=md_ev[:-1]; dm=np.maximum(md_ev-prev,1.0)
    kap=np.asarray(kap_v_ev,np.float64)
    if kap.shape[0]==len(hw): kap=kap[ev_pos]
    rk=np.asarray(rk_full,np.float64)[ev_pos]
    return {"n":ev_pos.size,"out":out,"ev_pos":ev_pos,"z":z_all[ev_pos],"gr":gr_all[ev_pos],"rk":rk,"kap":kap,"dm":dm,"init_pos":last_tvt+last_Z,"init_rate":ir}

def run_pf_wells_cuda_batch_perseed(jobs, NS, NP, base_seed, gs, FLL=1000, vn=0.002, pn=0.002,
                            device="cuda", dtype=torch.float32, resample_row_chunk=64,
                            emission="gauss", seg_W=80, seg_c=10.0):
    SEG = (emission == "segcorr"); seg_W = int(seg_W)
    results=[None]*len(jobs); prepped=[]
    for j,job in enumerate(jobs):
        tg_mean=float(np.mean(job["tw_gr"])); p=_prep(job["hw"],job["rk_full"],job["kap_v_ev"],tg_mean)
        p["orig_j"]=j; p["tw_tvt"]=np.asarray(job["tw_tvt"],np.float64); p["tw_gr"]=np.asarray(job["tw_gr"],np.float64)
        results[j]={"out":p["out"],"ev_pos":p["ev_pos"],"seed_preds":np.zeros((NS,p["n"])),"logz":np.zeros(NS)}
        if p["n"]>0: prepped.append(p)
    if not prepped: return results
    prepped.sort(key=lambda x:x["n"],reverse=True)
    B=len(prepped); n_ev=np.array([p["n"] for p in prepped]); max_n=int(n_ev[0]); Mmax=max(len(p["tw_tvt"]) for p in prepped)
    z_np=np.zeros((B,max_n)); gr_np=np.zeros((B,max_n)); rk_np=np.zeros((B,max_n)); kap_np=np.zeros((B,max_n)); dm_np=np.zeros((B,max_n))
    TW=np.zeros((B,Mmax)); TG=np.zeros((B,Mmax)); Ml=np.zeros(B,np.int64); ip=np.empty(B); ir=np.empty(B)
    for b,p in enumerate(prepped):
        n=p["n"]; z_np[b,:n]=p["z"]; gr_np[b,:n]=p["gr"]; rk_np[b,:n]=p["rk"]; kap_np[b,:n]=p["kap"]; dm_np[b,:n]=p["dm"]
        m=len(p["tw_tvt"]); Ml[b]=m; TW[b,:m]=p["tw_tvt"]; TW[b,m:]=p["tw_tvt"][-1]+1e9; TG[b,:m]=p["tw_gr"]; TG[b,m:]=p["tw_gr"][-1]
        ip[b]=p["init_pos"]; ir[b]=p["init_rate"]
    with torch.no_grad():
        tt=lambda a,d=dtype: torch.as_tensor(a,device=device,dtype=d)
        z_v=tt(z_np); gr_v=tt(gr_np); rk_v=tt(rk_np); kap_v=tt(kap_np); dm_v=tt(dm_np); TWt=tt(TW); TGt=tt(TG)
        Ml_t=tt(Ml,torch.long); ar_b=torch.arange(B,device=device)
        tmin_b=TWt[:,0]; tmax_b=TWt[ar_b,Ml_t-1]
        gen=torch.Generator(device=device); gen.manual_seed(int(base_seed))
        pos=tt(ip)[:,None,None]+2.0*torch.randn((B,NS,NP),device=device,dtype=dtype,generator=gen)
        rate=tt(ir)[:,None,None]+0.01*torch.randn((B,NS,NP),device=device,dtype=dtype,generator=gen)
        w=torch.full((B,NS,NP),1.0/NP,device=device,dtype=dtype)
        buf=torch.empty((B,NS,FLL,NP),device=device,dtype=dtype) if FLL>0 else None
        segbuf=torch.empty((B,NS,seg_W,NP),device=device,dtype=dtype) if SEG else None
        obsbuf=torch.empty((B,seg_W),device=device,dtype=dtype) if SEG else None
        pred=torch.zeros((B,NS,max_n),device=device,dtype=dtype); logz=torch.zeros((B,NS),device=device,dtype=torch.float32)
        ar=torch.arange(NP,device=device,dtype=dtype).view(1,NP)/NP
        A=B
        for i in range(max_n):
            while A>0 and n_ev[A-1]<=i: A-=1
            if A==0: break
            pa=pos[:A]; ra=rate[:A]; wa=w[:A]; zi=z_v[:A,i].view(A,1,1); gri=gr_v[:A,i].view(A,1,1)
            rki=rk_v[:A,i].view(A,1,1); kpi=kap_v[:A,i].view(A,1,1); dmi=dm_v[:A,i].view(A,1,1)
            tmn=tmin_b[:A].view(A,1,1); tmx=tmax_b[:A].view(A,1,1)
            ra+=kpi*(rki-ra)+vn*torch.randn((A,NS,NP),device=device,dtype=dtype,generator=gen)
            pa+=ra*dmi+pn*torch.randn((A,NS,NP),device=device,dtype=dtype,generator=gen)
            tvt_p=torch.clamp(pa-zi,min=tmn-100.0,max=tmx+100.0); pa.copy_(tvt_p+zi)
            if not SEG:
                q=torch.clamp(tvt_p,min=tmn,max=tmx).reshape(A,NS*NP)
                i1=torch.searchsorted(TWt[:A].contiguous(),q.contiguous(),right=False)
                i1=torch.minimum(i1.clamp_(min=1),(Ml_t[:A]-1).view(A,1)); i0=i1-1
                x0=torch.gather(TWt[:A],1,i0); x1=torch.gather(TWt[:A],1,i1); y0=torch.gather(TGt[:A],1,i0); y1=torch.gather(TGt[:A],1,i1)
                fr=(q-x0)/torch.clamp(x1-x0,min=1e-12); eg=(y0*(1.0-fr)+y1*fr).reshape(A,NS,NP)
                d=(gri-eg)/gs; loglk=-0.5*torch.clamp(d*d,max=600.0)
                mxg=loglk.max(dim=2,keepdim=True).values; loglk-=mxg
                wa*=torch.exp(loglk); sg=wa.sum(dim=2,keepdim=True)
                logz[:A]+=(mxg.squeeze(2)+torch.log(sg.squeeze(2)+1e-300)).float()
                wa/=sg; updated=True
            else:
                bp=i%seg_W; segbuf[:A,:,bp,:]=tvt_p; obsbuf[:A,bp]=gr_v[:A,i]; updated=(bp==seg_W-1)
                if updated:
                    win=segbuf[:A]
                    qq=torch.clamp(win,min=tmn.view(A,1,1,1),max=tmx.view(A,1,1,1)).reshape(A,NS*seg_W*NP)
                    j1=torch.searchsorted(TWt[:A].contiguous(),qq.contiguous(),right=False)
                    j1=torch.minimum(j1.clamp_(min=1),(Ml_t[:A]-1).view(A,1)); j0=j1-1
                    x0=torch.gather(TWt[:A],1,j0); x1=torch.gather(TWt[:A],1,j1); y0=torch.gather(TGt[:A],1,j0); y1=torch.gather(TGt[:A],1,j1)
                    fr=(qq-x0)/torch.clamp(x1-x0,min=1e-12); egw=(y0*(1.0-fr)+y1*fr).reshape(A,NS,seg_W,NP)
                    obs=obsbuf[:A]; o=obs-obs.mean(dim=1,keepdim=True)
                    e=egw-egw.mean(dim=2,keepdim=True)
                    num=(o.view(A,1,seg_W,1)*e).sum(dim=2)
                    o2=(o*o).sum(dim=1).view(A,1,1); e2=(e*e).sum(dim=2)
                    corr=num/torch.sqrt(o2*e2+1e-9)
                    loglk=seg_c*corr
                    mxs=loglk.max(dim=2,keepdim=True).values; loglk-=mxs
                    wa*=torch.exp(loglk); ss=wa.sum(dim=2,keepdim=True)
                    logz[:A]+=(mxs.squeeze(2)+torch.log(ss.squeeze(2)+1e-300)).float()
                    wa/=ss
            if FLL>0:
                slot=i%FLL; ba=buf[:A]
                if i>=FLL: pred[:A,:,i-FLL]=(wa*ba[:,:,slot,:]).sum(dim=2)-z_v[:A,i-FLL].view(A,1)
                ba[:,:,slot,:]=pa
            else: pred[:A,:,i]=(wa*(pa-zi)).sum(dim=2)
            if updated:
                ess=1.0/(wa*wa).sum(dim=2); need=torch.nonzero(ess<0.5*NP,as_tuple=False)
                if need.numel()>0:
                    rows=need[:,0]*NS+need[:,1]; pf=pa.reshape(A*NS,NP); rf=ra.reshape(A*NS,NP); wf=wa.reshape(A*NS,NP)
                    wn=wf.index_select(0,rows); cum=torch.cumsum(wn,1); cum[:,-1]=1.0
                    u0=torch.rand((rows.numel(),1),device=device,dtype=dtype,generator=gen)/NP; idx=torch.searchsorted(cum,u0+ar,right=False).clamp_(max=NP-1)
                    pf[rows]=torch.gather(pf.index_select(0,rows),1,idx)+0.1*torch.randn((rows.numel(),NP),device=device,dtype=dtype,generator=gen)
                    rf[rows]=torch.gather(rf.index_select(0,rows),1,idx)+0.001*torch.randn((rows.numel(),NP),device=device,dtype=dtype,generator=gen)
                    wf[rows]=1.0/NP
                    if FLL>0:
                        bf=ba.reshape(A*NS,FLL,NP); m=int(rows.numel())
                        for st in range(0,m,resample_row_chunk):
                            en=min(st+resample_row_chunk,m); r=rows[st:en]; ic=idx[st:en]
                            bf[r]=torch.gather(bf.index_select(0,r),2,ic[:,None,:].expand(-1,FLL,-1))
        if FLL>0:
            for b,p in enumerate(prepped):
                n=p["n"]; start=max(0,n-FLL)
                if n<=start: continue
                slots=torch.arange(start,n,device=device)%FLL
                tb=buf[b,:,slots,:].permute(1,0,2); vals=(tb*w[b][None]).sum(dim=2)
                pred[b,:,start:n]=vals.permute(1,0)-z_v[b,start:n].view(1,-1)
        pred_cpu=pred.detach().cpu().numpy(); logz_cpu=logz.detach().cpu().numpy()
    for b,p in enumerate(prepped):
        r=results[p["orig_j"]]; r["seed_preds"]=pred_cpu[b,:,:p["n"]].copy(); r["logz"]=logz_cpu[b].copy()
    return results
