"""Резидуальный 2D-UNet (порт GeoSteerNet medali1992) + proximity-weighted masked MSE.

Вход: 2 канала [heatmap, history] формы [B,1,64,24] каждый -> [B,2,64,24].
Выход: SDF [B,1,64,24] (голова без активации, безграничный EDT-SDF).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pad=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.proj = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.proj(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.res(x)
        return skip, self.pool(skip)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))

class CorrStem(nn.Module):
    """XVOL (research/RADICAL_ARCHITECTURE_RESEARCH.md, "runner-up"): обучаемый матч-объём
    (RAFT/IGEV-style) поверх существующих misfit/NCC-примитивов render_sw.py. q_enc кодирует СЫРОЙ
    (native-res, S_SW точек/столбец, h_gr_rawi) GR каждого столбца -> f_h [B,H,d]; t_enc кодирует
    typewell GR на строчной (T_STEP-сетке) -> f_t [B,T,d]; корреляция ОДНИМ einsum'ом (не Python-циклом
    по лагам, как USE_NCC) даёт C0[B,T,H] — той же формы, что остальные image-каналы -> конкатенируется
    напрямую в GeoSteerNet.forward() ПЕРЕД self.stem(). 3-уровневая пирамида — avg_pool/upsample C0 по
    T (offset-ось), H (ось столбцов/query) не пулим — факторами /2,/4, БЕЗ доп. параметров.

    d=32 фичи, n_scales=3 уровня пирамиды. q_enc ЖЁСТКО рассчитан на s_native=32 (=S_SW): 5x stride-2
    (32->16->8->4->2->1) — при изменении ROGII_SW_S архитектуру пересчитывать заново (assert в forward
    страхует от тихого мисматча). t_enc — дилатированный стек (dilation 1,2,4,8,16), stride=1, padding
    держит длину РАВНОЙ входной для любого T."""

    def __init__(self, s_native: int = 32, d: int = 32, n_scales: int = 3):
        super().__init__()
        self.s_native, self.d, self.n_scales = s_native, d, n_scales
        self.q_enc = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(16, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(32, d, 2, stride=2, padding=0),
        )
        self.t_enc = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(16, 24, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(24, 32, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=16, dilation=8), nn.GELU(),
            nn.Conv1d(32, d, 3, padding=16, dilation=16),
        )

    def forward(self, h_native: torch.Tensor, t_native: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        """h_native [B,H,S] (H=H_MAX_SW, S=s_native — render_sw xvol_h_native: сырое (h_gr_rawi) окно
        по столбцу, паддинг нулями за n[b]). t_native [B,T] (render_sw xvol_t_native = t_seg_gr, ТОТ ЖЕ
        массив, что уже broadcast в существующий ch0). n_valid [B] long = "n" из build_chunk_sw (сколько
        столбцов реальны). Возврат [B, n_scales, T, H] — готово для torch.cat([x, corr_stem(...)], dim=1)
        ПЕРЕД self.stem(x) в GeoSteerNet.forward()."""
        B, H, S = h_native.shape
        assert S == self.s_native, f"CorrStem: q_enc жёстко под s_native={self.s_native}, получено {S}"
        T = t_native.shape[1]
        f_h = F.normalize(self.q_enc(h_native.reshape(B * H, 1, S) / 100.0).reshape(B, H, self.d), dim=-1)
        f_t = F.normalize(self.t_enc((t_native / 100.0).unsqueeze(1)).transpose(1, 2), dim=-1)
        C0 = torch.einsum('btd,bhd->bth', f_t, f_h)
        levels = [C0]
        cur = C0.transpose(1, 2)
        for _ in range(self.n_scales - 1):
            cur = F.avg_pool1d(cur, kernel_size=2, stride=2)
            levels.append(F.interpolate(cur, size=T, mode="nearest").transpose(1, 2))
        out = torch.stack(levels, dim=1)
        idx = torch.arange(H, device=h_native.device)
        mask = (idx[None, :] < n_valid[:, None]).to(out.dtype)[:, None, None, :]
        return out * mask

class ParamHeads(nn.Module):
    """exp_201 K7-PARAM-WTA: три ПАРАМЕТРИЧЕСКИХ кандидата ПОВЕРХ нетронутых K=4 abs-мод.
    Каждая голова предсказывает per-column приращение пути (ft/столбец) из masked-GAP(T)
    фич up1, разворачивает его направленным cumsum от последнего известного столбца (nh-1)
    в путь и кладёт в SDF-поле ТОЙ ЖЕ формулой, что GT-таргет build_chunk_sw:
    clip((path - t_seg_tvt)/SDF_SCALE, ±clamp). Головы: drift (гибкий дрейф ±max_step/кол),
    planar (кусочно-линейный, изломы каждые seg_cols столбцов — против дребезга E4),
    fault (спящий гейтованный скачок до ±max_jump — сигнатура E1 "один шаг + смещение").
    НОЛЬ новых лоссов: кандидаты входят в mtp_mix_loss (hindsight best-of-K = WTA) и в
    soft_reanchor/rollout наравне с abs-модами (все три K-agnostic по форме)."""

    def __init__(self, b: int, sdf_scale: float, clamp: float,
                 max_step_ft: float = 8.0, max_jump_ft: float = 60.0, seg_cols: int = 4):
        super().__init__()
        self.scale, self.clamp = float(sdf_scale), float(clamp)
        self.max_step, self.max_jump, self.seg = float(max_step_ft), float(max_jump_ft), int(seg_cols)

        def _stack(out_ch):
            return nn.Sequential(nn.Conv1d(b, b, 5, padding=2), nn.ReLU(inplace=True),
                                 nn.Conv1d(b, b, 5, padding=4, dilation=2), nn.ReLU(inplace=True),
                                 nn.Conv1d(b, out_ch, 1))
        self.drift = _stack(1)
        self.planar = _stack(1)
        self.fault = _stack(3)
        self.logit_p = nn.Conv2d(b, 3, 1)
        for m in (self.drift[-1], self.planar[-1], self.fault[-1], self.logit_p):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.fault[-1].bias[1] = -4.0
            self.logit_p.bias.fill_(-2.0)

    def forward(self, x, t_mask, h_mask, t_seg_tvt):
        """x [B,b,T,H] (up1-фичи), h_mask [B,H], t_seg_tvt [B,T]. -> sdf_p [B,3,T,H],
        logit_p [B,3,H], p_gate [B,H]."""
        B, C, T, H = x.shape
        if t_mask is not None:
            tw = t_mask[:, None, :, None].to(x.dtype)
            xt = (x * tw).sum(2) / (tw.sum(2) + 1e-6)
        else:
            xt = x.mean(2)

        nh1 = (torch.argmax((h_mask > 0).to(x.dtype), 1) - 1).clamp(min=0)
        anchor = t_seg_tvt[:, T // 2][:, None]

        def _unroll(delta):
            cs = torch.cumsum(delta, 1)
            return anchor + cs - cs.gather(1, nh1[:, None])

        d_dr = torch.tanh(self.drift(xt)[:, 0]) * self.max_step
        pl = torch.tanh(self.planar(F.avg_pool1d(xt, self.seg, self.seg))[:, 0]) * self.max_step
        d_pl = pl.repeat_interleave(self.seg, 1)
        if d_pl.shape[1] < H:
            d_pl = torch.cat([d_pl, d_pl[:, -1:].expand(-1, H - d_pl.shape[1])], 1)
        fa = self.fault(xt)
        p_gate = torch.sigmoid(fa[:, 1])
        d_fa = torch.tanh(fa[:, 0]) * 2.0 + p_gate * torch.tanh(fa[:, 2]) * self.max_jump
        fields = [torch.clamp((_unroll(d)[:, None, :] - t_seg_tvt[:, :, None]) / self.scale,
                              -self.clamp, self.clamp) for d in (d_dr, d_pl, d_fa)]
        return torch.stack(fields, 1), self.logit_p(x).mean(2), p_gate

class EditorHead(nn.Module):
    """exp_203 EDIT-BULK: ПОСЛЕДОВАТЕЛЬНАЯ (не соревновательная) поправка к УЖЕ декодированному пути.
    Урок exp_201 (W6): голова, попавшая в один WTA/min_k-слот с abs-модами, монополизирует градиент
    и убивает чемпиона. Здесь голова НЕ участвует ни в perc.min, ни в ce_mix, ни в soft_reanchor —
    у неё свой лосс, свой градиент, и она читает x.detach() (жёстко, не опция).
    Выход: edit_delta [B,H] в футах, аддитивно к col_tvt в rollout (edit_gain=0 -> байт-в-байт чемпион).
    Структурное ограничение seg (avg_pool -> repeat_interleave): голова не может колебаться быстрее
    seg столбцов (~192ft при seg=6) — архитектурная (не лоссовая) атака на дребезг E4."""

    def __init__(self, b: int, cap_ft: float = 8.0, seg: int = 6):
        super().__init__()
        self.cap, self.seg = float(cap_ft), int(seg)
        self.net = nn.Sequential(
            nn.Conv1d(b, b, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(b, b, 5, padding=4, dilation=2), nn.ReLU(inplace=True),
            nn.Conv1d(b, 1, 1))
        nn.init.normal_(self.net[-1].weight, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t_mask):
        x = x.detach()
        if t_mask is not None:
            tw = t_mask[:, None, :, None].to(x.dtype)
            xt = (x * tw).sum(2) / (tw.sum(2) + 1e-6)
        else:
            xt = x.mean(2)
        H = xt.shape[2]
        d = self.net(F.avg_pool1d(xt, self.seg, self.seg))[:, 0]
        d = d.repeat_interleave(self.seg, 1)
        if d.shape[1] < H:
            d = torch.cat([d, d[:, -1:].expand(-1, H - d.shape[1])], 1)
        return torch.tanh(d[:, :H]) * self.cap

def decode_rows_fb(sdf_sel, h_mask, tau=0.2, lam=1.1, K=4):
    """Деплой-matched FB posterior-mean строки для КАЖДОГО столбца (torch-твин recover_tvt_soft).
    Механика идентична soft_fb_loss, но возвращает mean_row [B,H] вместо лосса. sdf_sel [B,T,H]."""
    B, T, H = sdf_sel.shape
    phi = -sdf_sel.abs() / max(tau, 1e-6)
    idx = torch.arange(H, device=sdf_sel.device, dtype=torch.float32)
    last = (h_mask * idx[None, :]).max(1).values
    colvalid = (idx[None, :] <= last[:, None]).to(phi.dtype)
    phi = phi * colvalid[:, None, :]
    la = [phi[:, :, 0]]
    for c in range(1, H):
        la.append(phi[:, :, c] + _band_lse_t(la[-1], K, lam))
    lb = [torch.zeros_like(phi[:, :, 0])]
    for c in range(H - 2, -1, -1):
        lb.append(_band_lse_t(phi[:, :, c + 1] + lb[-1], K, lam, bwd=True))
    logg = torch.stack(la, 2) + torch.stack(lb[::-1], 2)
    gamma = torch.softmax(logg, dim=1)
    rows = torch.arange(T, device=sdf_sel.device, dtype=sdf_sel.dtype)
    return (gamma * rows[None, :, None]).sum(1)

def _rows_to_ft(rows, t_seg_tvt):
    """Дробная строка [B,H] -> футы через линейную интерполяцию по сетке t_seg_tvt [B,T]."""
    T = t_seg_tvt.shape[1]
    r = rows.clamp(0, T - 1 - 1e-4)
    lo = r.floor().long()
    w = (r - lo.to(r.dtype))
    a = t_seg_tvt.gather(1, lo)
    bb = t_seg_tvt.gather(1, (lo + 1).clamp(max=T - 1))
    return a + w * (bb - a)

def editor_loss(sdf_abs, logit_abs, delta_pred, true, h_mask, t_seg_tvt,
                dmax_ft=15.0, tau=0.2, lam=1.1, K=4, huber=3.0):
    """exp_203: голова учится предсказывать ОСТАТОК деплойного декода чемпиона (не поле!).
    Вся референс-часть под no_grad => это константа, abs-моды/селектор не получают отсюда градиента
    (W6-safe). Маска |err|<dmax вырезает branch-катастрофы из супервизии: их сигнал неизвлекаем (W1),
    и без маски они доминировали бы градиент. Huber — линейный хвост."""
    with torch.no_grad():
        k_star = logit_abs.argmax(1)
        idx = k_star[:, None, None, :].expand(-1, -1, sdf_abs.shape[2], -1)
        sdf_sel = sdf_abs.gather(1, idx).squeeze(1)
        col_ft = _rows_to_ft(decode_rows_fb(sdf_sel, h_mask, tau, lam, K), t_seg_tvt)
        gt_ft = _rows_to_ft(true.squeeze(1).abs().argmin(1).to(col_ft.dtype), t_seg_tvt)
        err = gt_ft - col_ft
        m = (h_mask > 0).to(err.dtype) * (err.abs() < dmax_ft).to(err.dtype)
    ae = (delta_pred - err).abs()
    l = torch.where(ae <= huber, 0.5 * ae ** 2, huber * (ae - 0.5 * huber))
    loss = (l * m).sum() / (m.sum() + 1e-8) / (dmax_ft ** 2)
    return loss, err.detach(), m.detach()

class GeoSteerNet(nn.Module):
    """2-канальный резид-UNet: (64,24)->(32,12)->(16,6)->(8,3)->декодер->SDF [B,1,64,24]."""

    def __init__(self, base_ch: int = 32, in_ch: int = 2, out_clamp: float = None, prox_tau: float = 5.0,
                 n_modes: int = 1, slope_w: float = 0.0, soft_ft_w: float = 0.0, ce_w: float = 0.0,
                 fb_w: float = 0.0, fb_tau: float = 0.2, fb_lam: float = 1.1, fb_K: int = 4,
                 dz_w: float = 0.0, strat_w: float = 0.0, strat_tau: float = 15.0, strat_detach: bool = False,
                 xvol: bool = False, xvol_d: int = 32, xvol_scales: int = 3,
                 param_heads: bool = False, sdf_scale: float = 50.0,
                 edit_head: bool = False, edit_cap_ft: float = 8.0, edit_seg: int = 6):
        super().__init__()
        b = base_ch
        self.dz_w = dz_w
        self.strat_w = strat_w

        self.strat_tau = strat_tau
        self.strat_detach = strat_detach
        self.out_clamp = out_clamp
        self.prox_tau = prox_tau
        self.slope_w = slope_w
        self.soft_ft_w = soft_ft_w
        self.ce_w = ce_w

        self.fb_w = fb_w
        self.fb_tau, self.fb_lam, self.fb_K = fb_tau, fb_lam, fb_K
        self.pure_ce = False
        self.n_modes = n_modes
        self.xvol = xvol
        stem_in = in_ch + (xvol_scales if xvol else 0)
        if xvol:
            self.corr_stem = CorrStem(d=xvol_d, n_scales=xvol_scales)
        self.stem = ConvBnRelu(stem_in, b, k=5, pad=2)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        self.bottleneck = nn.Sequential(ResidualBlock(b * 8, b * 8), ResidualBlock(b * 8, b * 8))
        self.up3 = UpBlock(b * 8 + b * 8, b * 4)
        self.up2 = UpBlock(b * 4 + b * 4, b * 2)
        self.up1 = UpBlock(b * 2 + b * 2, b)
        self.sdf_head = nn.Conv2d(b, n_modes, 1)
        if n_modes > 1:
            self.logit_head = nn.Conv2d(b, n_modes, 1)
        if ce_w > 0:
            self.row_head = nn.Sequential(
                nn.Conv2d(b, b, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(b, 1, 1))
        if dz_w > 0:
            self.dz_head = nn.Sequential(nn.Linear(2 * b, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        if strat_w > 0:
            self.struct_head = nn.Conv2d(b, 1, 1)
        self._init_weights()
        if dz_w > 0:
            nn.init.zeros_(self.dz_head[-1].weight)
            nn.init.zeros_(self.dz_head[-1].bias)
        if strat_w > 0:
            nn.init.normal_(self.struct_head.weight, std=0.01)
            nn.init.zeros_(self.struct_head.bias)

        self.param_heads = param_heads
        if param_heads:
            assert n_modes > 1, "--param-heads требует MTP (n_modes>1)"
            self.pheads = ParamHeads(b, sdf_scale, out_clamp if out_clamp else 3.0)
        self.edit_head = edit_head
        if edit_head:
            self.ehead = EditorHead(b, edit_cap_ft, edit_seg)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.sdf_head.weight, std=0.01)
        nn.init.zeros_(self.sdf_head.bias)

    def forward(self, batch: dict) -> dict:
        x = batch["image"] if "image" in batch else torch.cat([batch["heatmap"], batch["history"]], dim=1)
        if self.xvol:
            corr = self.corr_stem(batch["xvol_h"], batch["xvol_t"], batch["xvol_n"])
            x = torch.cat([x, corr], dim=1)
        x = self.stem(x)
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        sdf = self.sdf_head(x)
        if self.out_clamp:
            sdf = torch.tanh(sdf) * self.out_clamp
        out = {"sdf": sdf}
        if self.dz_w > 0:
            out["dz"] = self._dz_forward(x, batch)
        if self.strat_w > 0:
            out["struct"] = self.struct_head(x.detach() if self.strat_detach else x)
        if self.n_modes > 1:
            out["logit"] = self.logit_head(x).mean(2)
            if getattr(self, "param_heads", False):
                sdf_p, logit_p, p_gate = self.pheads(x, batch.get("t_mask"),
                                                     batch["h_mask"], batch["t_seg_tvt"])
                sdf = torch.cat([sdf, sdf_p], 1)
                out["sdf"] = sdf
                out["logit"] = torch.cat([out["logit"], logit_p], 1)
                out["p_gate"] = p_gate
        if getattr(self, "edit_head", False):
            out["edit_delta"] = self.ehead(x, batch.get("t_mask"))
        if self.ce_w > 0:
            out["rowlogit"] = self.row_head(x)
        if "sdf" in batch:
            if self.pure_ce:
                out["sdf_loss"] = costmap_ce(sdf, batch["sdf"], batch["h_mask"], batch["t_mask"])
            elif self.n_modes > 1 and getattr(self, "mtp_mix", False):
                out["sdf_loss"] = mtp_mix_loss(sdf, out["logit"], batch["sdf"], batch["h_mask"],
                                               batch["t_mask"], tau=self.prox_tau)
            elif self.n_modes > 1:
                out["sdf_loss"] = mtp_loss(sdf, out["logit"], batch["sdf"], batch["h_mask"],
                                           batch["t_mask"], tau=self.prox_tau)
            else:
                out["sdf_loss"] = proximity_masked_mse(sdf, batch["sdf"], batch["h_mask"],
                                                       batch["t_mask"], tau=self.prox_tau)
                if self.slope_w > 0:
                    out["sdf_loss"] = out["sdf_loss"] + self.slope_w * slope_consistency(
                        sdf, batch["sdf"], batch["h_mask"], batch["t_mask"], tau=self.prox_tau)
                if self.soft_ft_w > 0:
                    out["sdf_loss"] = out["sdf_loss"] + self.soft_ft_w * soft_decode_loss(
                        sdf, batch["sdf"], batch["h_mask"], batch["t_mask"])
                if self.ce_w > 0:
                    out["sdf_loss"] = out["sdf_loss"] + self.ce_w * costmap_ce(
                        out["rowlogit"], batch["sdf"], batch["h_mask"], batch["t_mask"])
                if self.fb_w > 0:
                    out["sdf_loss"] = out["sdf_loss"] + self.fb_w * soft_fb_loss(
                        sdf, batch["sdf"], batch["h_mask"], batch["t_mask"],
                        tau=self.fb_tau, lam=self.fb_lam, K=self.fb_K)
            if self.strat_w > 0 and "sdf_struct" in batch:
                sw_mask = batch["hc_mask"] * batch["struct_valid"][:, None]
                out["sdf_loss"] = out["sdf_loss"] + self.strat_w * proximity_masked_mse(
                    out["struct"], batch["sdf_struct"], sw_mask, batch["t_mask"], tau=self.strat_tau)
        return out

    def _dz_forward(self, x, batch):
        """P2 δ̂-head: x = фичи up1 [B,b,T,H]. Маскированный GAP по T (t_mask) и по предсказываемым
        столбцам (h_mask>0) + отдельный токен = GAP по hist-столбцам (до первого предсказываемого);
        concat -> MLP(2b→64→2) -> (δ̂, log σ̂). Без h_mask в batch — фолбэк на все столбцы."""
        B, C, T, H = x.shape
        tm = batch.get("t_mask")
        if tm is not None:
            tw = tm[:, None, :, None].to(x.dtype)
            xt = (x * tw).sum(2) / (tw.sum(2) + 1e-6)
        else:
            xt = x.mean(2)
        hm = batch.get("h_mask")
        if hm is None:
            hm = torch.ones(B, H, device=x.device, dtype=x.dtype)
        pred_m = (hm > 0).to(x.dtype)
        idx = torch.arange(H, device=x.device)
        first = torch.argmax(pred_m, 1)
        hist_m = (idx[None, :] < first[:, None]).to(x.dtype)
        def _gap(m):
            return (xt * m[:, None, :]).sum(2) / (m.sum(1, keepdim=True) + 1e-6)
        tok = torch.cat([_gap(pred_m), _gap(hist_m)], 1)
        return self.dz_head(tok)

def slope_consistency(pred, true, h_mask, t_mask, tau: float = 0.5):
    """MSE на разности соседних столбцов (наклон матч-линии). Ramp-ошибка = накопленный
    сдвиг наклона; явная супервизия d(sdf)/dMD давит систематический недолёт тренда."""
    dp = pred[..., 1:] - pred[..., :-1]
    dt = true[..., 1:] - true[..., :-1]
    m = (h_mask[:, None, None, 1:] * h_mask[:, None, None, :-1]) * t_mask[:, None, :, None]
    w = m * torch.exp(-torch.abs(true[..., 1:]) / tau)
    return (w * (dp - dt) ** 2).sum() / (w.sum() + 1e-8)

def proximity_masked_mse(pred, true, h_mask, t_mask, tau: float = 5.0):
    """MSE, взвешенный exp(-|sdf_true|/tau) (фокус у нуль-кроссинга) и маской t⊗h."""
    mask = h_mask[:, None, None, :] * t_mask[:, None, :, None]
    w = mask * torch.exp(-torch.abs(true) / tau)
    return (w * (pred - true) ** 2).sum() / (w.sum() + 1e-8)

def _band_lse_t(la, K, lam, bwd=False):
    """logsumexp_{d=-K..K}(la[shift d] - lam|d|) по строкам (dim=1), torch, дифф. bwd=обратный сдвиг.
    Векторизовано по d (было: 9 последовательных torch.roll+clone на Python-цикле — доминирующий
    бутылочное горлышко в soft_reanchor/UNROLL-2, профилировано 2026-07-15: один вызов дороже целого
    baseline train-step). torch.gather по одной индекс-матрице [T,2K+1] вместо K roll-ов; численно
    идентично оригиналу (maxdiff ~1e-7, float-шум), ~1.8x быстрее на вызов."""
    T = la.shape[1]
    NEG = -1e9
    ds = torch.arange(-K, K + 1, device=la.device)
    sd = -ds if bwd else ds
    t = torch.arange(T, device=la.device)
    src = t[:, None] - sd[None, :]
    valid = (src >= 0) & (src < T)
    src_c = src.clamp(0, T - 1)
    gathered = la[:, src_c]
    pen = lam * ds.abs().to(la.dtype)[None, None, :]
    out = torch.where(valid[None, :, :], gathered - pen, torch.full_like(gathered, NEG))
    return torch.logsumexp(out, dim=2)

def costmap_ce(logits, true, h_mask, t_mask, sigma_label=3.0):
    """SegN-v2 Stage 1: CE per column против гаусс-размытой GT-строки. logits [B,1,T,H] БЕЗ tanh
    (out_clamp=None), true = наш sdf-таргет (нуль-кроссинг = GT-строка). Softmax по строкам даёт
    КАЛИБРОВАННЫЙ постериор для posterior-mean декода (наш |sdf|-прокси не калиброван)."""
    B, _, T, H = logits.shape
    lg = logits.squeeze(1)
    with torch.no_grad():
        gt_row = true.squeeze(1).abs().argmin(1)
        rows = torch.arange(T, device=logits.device, dtype=logits.dtype)
        tgt = torch.exp(-0.5 * ((rows[None, :, None] - gt_row[:, None, :].to(logits.dtype)) / sigma_label) ** 2)
        tgt = tgt * t_mask[:, :, None]
        tgt = tgt / (tgt.sum(1, keepdim=True) + 1e-9)
    logp = torch.log_softmax(lg, dim=1)
    ce = -(tgt * logp).sum(1)
    colmask = (h_mask > 0).float()
    return (ce * colmask).sum() / (colmask.sum() + 1e-8)

def soft_fb_loss(sdf, true, h_mask, t_mask, tau=0.2, lam=1.1, K=4, delta=3.0):
    """[7.5] deploy-matched: дифф. FB posterior-mean ЧЕРЕЗ band-переходы = ТОЧНО деплойный
    recover_tvt_soft (в отличие от строумена #7 = per-column softmax без переходов).
    Huber(mean_row - gt_row, delta=3 строки) на eval-столбцах: линейный хвост не даёт
    катастрофам доминировать градиент (clean-transfer философия). phi за последним валидным
    столбцом обнуляется (плоская эмиссия ~ деплойный конец цепи lb[n-1]=0)."""
    B, _, T, H = sdf.shape
    phi = -sdf.squeeze(1).abs() / max(tau, 1e-6)
    with torch.no_grad():
        idx = torch.arange(H, device=sdf.device, dtype=torch.float32)
        last = (h_mask * idx[None, :]).max(1).values
        colvalid = (idx[None, :] <= last[:, None]).float()
    phi = phi * colvalid[:, None, :]
    la = [phi[:, :, 0]]
    for c in range(1, H):
        la.append(phi[:, :, c] + _band_lse_t(la[-1], K, lam))
    lb = [torch.zeros_like(phi[:, :, 0])]
    for c in range(H - 2, -1, -1):
        lb.append(_band_lse_t(phi[:, :, c + 1] + lb[-1], K, lam, bwd=True))
    logg = torch.stack(la, 2) + torch.stack(lb[::-1], 2)
    gamma = torch.softmax(logg, dim=1)
    rows = torch.arange(T, device=sdf.device, dtype=sdf.dtype)
    mean_row = (gamma * rows[None, :, None]).sum(1)
    with torch.no_grad():
        gt_row = true.squeeze(1).abs().argmin(1).to(sdf.dtype)
    colmask = (h_mask > 0).float()
    err = (mean_row - gt_row).abs()
    hub = torch.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    return (hub * colmask).sum() / (colmask.sum() + 1e-8)

def soft_decode_loss(sdf, true, h_mask, t_mask, tau=0.7):
    """#7: per-column soft-argmin — обучаем ПОЛЕ так, чтобы softmax-среднее по строке в каждом
    столбце попадало в GT-строку. Штрафует конкурирующую (неверную) моду, тянущую среднее в чужую
    ветку — прямая атака на census (сеть ставит массу на неверную моду). Векторизован, без FB-цикла.
    sdf/true [B,1,T,H]; вес exp(-|Δ|) фокусирует у нуль-кроссинга."""
    B, _, T, H = sdf.shape
    a = sdf.squeeze(1).abs()
    gamma = torch.softmax(-a / max(tau, 1e-6), dim=1)
    rows = torch.arange(T, device=sdf.device, dtype=sdf.dtype)
    mean_row = (gamma * rows[None, :, None]).sum(1)
    with torch.no_grad():
        gt_row = true.squeeze(1).abs().argmin(1).to(sdf.dtype)
    colmask = (h_mask > 0).float()
    return (colmask * (mean_row - gt_row) ** 2).sum() / (colmask.sum() + 1e-8)

def mtp_mix_loss(sdf_k, logit, true, h_mask, t_mask, tau=0.5, tau_e=0.5, mix_w=1.0, tv_w=0.3, sigma_label=3.0,
                 col_w=None, hist_w=0.0, hist_mask=None, dense_k=0.0, mcl_w=0.0, mcl_t=0.7):
    """MTP#2 (exp_121): селектор обучается ЧЕРЕЗ декод-распределение — mixture-постериор
    p(t|c) = sum_k w_k(c)*softmax_t(-|sdf_k|/tau_e), CE против гаусс-GT-строки; + best-of-K
    proximity (разнообразие веток) + TV на селектор. Градиент до logit течёт через микс."""
    B, K, T, H = sdf_k.shape
    logp_t = torch.log_softmax(-sdf_k.abs() / max(tau_e, 1e-6), dim=2)
    logw = torch.log_softmax(logit, dim=1)
    logmix = torch.logsumexp(logw[:, :, None, :] + logp_t, dim=1)
    with torch.no_grad():
        gt_row = true.squeeze(1).abs().argmin(1)
        rows = torch.arange(T, device=sdf_k.device, dtype=sdf_k.dtype)
        tgt = torch.exp(-0.5 * ((rows[None, :, None] - gt_row[:, None, :].to(sdf_k.dtype)) / sigma_label) ** 2)
        tgt = tgt * t_mask[:, :, None]
        tgt = tgt / (tgt.sum(1, keepdim=True) + 1e-9)
    colmask = (h_mask > 0).float()
    if col_w is not None:
        colmask = colmask * col_w
    if hist_mask is not None and hist_w > 0:
        colmask = colmask + hist_w * hist_mask
    ce_mix = (-(tgt * logmix).sum(1) * colmask).sum() / (colmask.sum() + 1e-8)
    w = ((colmask > 0).float()[:, None, None, :] * t_mask[:, None, :, None]) * torch.exp(-true.abs() / tau)
    perc = ((w * (sdf_k - true) ** 2).sum(2)) / (w.sum(2) + 1e-8)
    if dense_k > 0:

        with torch.no_grad():

            pc = perc - perc.min(1, keepdim=True).values
            scale = pc.max(1, keepdim=True).values.clamp_min(1e-6)
            r = torch.softmax(-(pc / scale) / max(dense_k, 1e-6), dim=1)
        sdf_l = ((r * perc).sum(1) * colmask).sum() / (colmask.sum() + 1e-8)
    else:
        sdf_l = (perc.min(1).values * colmask).sum() / (colmask.sum() + 1e-8)
    prob = torch.softmax(logit, 1)
    tv = (prob[:, :, 1:] - prob[:, :, :-1]).abs().sum(1)
    cm2 = colmask[:, 1:] * colmask[:, :-1]
    tv = (tv * cm2).sum() / (cm2.sum() + 1e-8)
    mcl = sdf_k.new_zeros(())
    if mcl_w > 0:

        ce_k = -(tgt[:, None] * logp_t).sum(2)
        with torch.no_grad():
            pc = ce_k - ce_k.min(1, keepdim=True).values
            sc = pc.max(1, keepdim=True).values.clamp_min(1e-6)
            r_k = torch.softmax(-(pc / sc) / max(mcl_t, 1e-6), dim=1)
        mcl = ((r_k * ce_k).sum(1) * colmask).sum() / (colmask.sum() + 1e-8)
    return sdf_l + mix_w * ce_mix + mcl_w * mcl + tv_w * tv

def soft_reanchor(sdf_k, logit, t_seg_tvt, rb_col, n_valid, tau=0.2, lam=1.1, K=4):
    """UNROLL-2 (exp_143): дифференцируемый ре-якорь — deploy-twin recover_tvt_soft на СМЕСИ мод.
    Эмиссия = mixture-постериор per column (как logmix в mtp_mix_loss: микс softmax(logit) по K,
    softmax_t(-|sdf_k|/tau) по строкам), FB band-переходы (_band_lse_t, деплойные lam/K),
    постериор-mean строки в колонке rb_col-1 (хвост-окно [rb-S_SW, rb) = ровно один столбец),
    линейная интерп в TVT по t_seg_tvt. БЕЗ detach: градиент течёт и в поле, и в селектор A.
    sdf_k [B,K,T,H], logit [B,K,H], t_seg_tvt [B,T], rb_col/n_valid [B] long. Возврат [B] ft.

    Усечённая FB-рекурсия (2026-07-16, профилирование по запросу юзера: backward доминировал
    train-step 59.8%): читаем logg только в ОДНОЙ колонке (j=rb_col-1) на сэмпл — la нужен лишь
    до max(j) по батчу, lb лишь от конца до min(j), а не по всему чанку. Математически ТОЧНО
    (не аппроксимация) то же самое, что полная рекурсия — проверено на реальных батчах:
    output/gradient maxdiff=0.0. Даёт 1.10x соло, 1.62x вместе с bf16-autocast (train_chunk.py)."""
    B, Km, T, H = sdf_k.shape
    n_max = int(n_valid.max())
    logp_t = torch.log_softmax(-sdf_k.abs() / max(tau, 1e-6), dim=2)
    logw = torch.log_softmax(logit, dim=1)
    phi = torch.logsumexp(logw[:, :, None, :] + logp_t, dim=1)
    idx = torch.arange(H, device=phi.device)
    colvalid = (idx[None, :] < n_valid[:, None]).to(phi.dtype)
    phi = (phi * colvalid[:, None, :])[:, :, :n_max]
    j = (rb_col - 1).clamp(0, n_max - 1)
    j_max = int(j.max().item()); j_min = int(j.min().item())
    la = [phi[:, :, 0]]
    for c in range(1, j_max + 1):
        la.append(phi[:, :, c] + _band_lse_t(la[-1], K, lam))
    lb = [torch.zeros_like(phi[:, :, 0])]
    for c in range(n_max - 2, j_min - 1, -1):
        lb.append(_band_lse_t(phi[:, :, c + 1] + lb[-1], K, lam, bwd=True))
    la_stack = torch.stack(la, 2)
    lb_stack = torch.stack(lb[::-1], 2)
    logg = la_stack[:, :, j_min:j_max + 1] + lb_stack[:, :, :j_max - j_min + 1]
    j_rel = (j - j_min).clamp(0, j_max - j_min)
    lg = logg.gather(2, j_rel[:, None, None].expand(B, T, 1)).squeeze(2)
    gamma = torch.softmax(lg, dim=1)
    rows = torch.arange(T, device=phi.device, dtype=phi.dtype)
    mean_row = (gamma * rows[None, :]).sum(1)
    lo = mean_row.detach().floor().long().clamp(0, T - 2)
    frac = (mean_row - lo.to(mean_row.dtype)).clamp(0, 1)
    tlo = t_seg_tvt.gather(1, lo[:, None]).squeeze(1)
    thi = t_seg_tvt.gather(1, (lo + 1)[:, None]).squeeze(1)
    return tlo * (1.0 - frac) + thi * frac

def conf_calib_loss(logit, rb_col, delta, pair_valid, margin=1.0):
    """exp_159 CONF-CALIB: калибрует энтропию K=4-селектора В КОЛОНКЕ ре-якоря против РЕАЛЬНОЙ
    (не model-estimated) ошибки ре-якоря delta = a_hat - atvt_b_true, которую УЖЕ считает UNROLL-2.
    Батч-попарный margin-ranking (без внешнего лейбла кроме delta): для пар (i,j) с |delta_i|-|delta_j|
    > margin штрафуем max(0, margin-(ent_i-ent_j)) — т.е. учим модель быть УВЕРЕННЕЕ (ниже энтропия)
    там, где ре-якорь оказался точнее, и НЕОПРЕДЕЛЁННЕЕ там, где реально ошиблась. Не циклично (delta —
    из teacher-forced GT, не из собственного роллаута — exp_156 DRIFT-JITTER ловушка), не даунвейт
    дальних/трудных чанков (exp_151 ловушка), реальный градиентный лосс на selector, не inference-
    time эвристика (Lattice v0 ловушка).
    logit [B,K,H], rb_col/pair_valid [B]. Возврат: скаляр loss, per-sample entropy [B] (для диагностики)."""
    B, K, H = logit.shape
    col = (rb_col - 1).clamp(0, H - 1)
    lg = logit.gather(2, col[:, None, None].expand(B, K, 1)).squeeze(2)
    p = torch.softmax(lg, dim=1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1)
    d = delta.detach().abs()
    diff_d = d[:, None] - d[None, :]
    diff_e = ent[:, None] - ent[None, :]
    pv = pair_valid[:, None] * pair_valid[None, :]
    sep = (diff_d > margin).float() * pv
    pen = torch.clamp(margin - diff_e, min=0.0) * sep
    n = sep.sum()
    loss = pen.sum() / n.clamp_min(1.0)
    return loss, ent

def unroll2_shift_ce(sdf_k, logit, true, h_mask, t_mask, shift_rows, valid, tau_e=0.5,
                     sigma_label=3.0, col_w=None):
    """UNROLL-2 (exp_143): mixture-CE чанка B против гаусс-таргета, сдвинутого ДИФФЕРЕНЦИРУЕМО на
    shift_rows [B] (= -Δ/T_STEP, Δ = â_A - atvt_B_true; знак: рост якоря => истинный TVT НИЖЕ по строкам).
    При Δ=0 == ce_mix из mtp_mix_loss (тот же гаусс-таргет). Градиент течёт в поле/селектор B И в Δ
    (=> в селектор/поле A через soft_reanchor). valid [B] маскирует пары без валидного чанка B."""
    B, K, T, H = sdf_k.shape
    logp_t = torch.log_softmax(-sdf_k.abs() / max(tau_e, 1e-6), dim=2)
    logw = torch.log_softmax(logit, dim=1)
    logmix = torch.logsumexp(logw[:, :, None, :] + logp_t, dim=1)
    with torch.no_grad():
        gt_row = true.squeeze(1).abs().argmin(1).to(sdf_k.dtype)
    rows = torch.arange(T, device=sdf_k.device, dtype=sdf_k.dtype)
    ctr = (gt_row + shift_rows[:, None]).clamp(0, T - 1)
    tgt = torch.exp(-0.5 * ((rows[None, :, None] - ctr[:, None, :]) / sigma_label) ** 2)
    tgt = tgt * t_mask[:, :, None]
    tgt = tgt / (tgt.sum(1, keepdim=True) + 1e-9)
    colmask = (h_mask > 0).float()
    if col_w is not None:
        colmask = colmask * col_w
    colmask = colmask * valid[:, None]
    ce = -(tgt * logmix).sum(1)
    return (ce * colmask).sum() / (colmask.sum() + 1e-8)

def mtp_loss(sdf_k, logit, true, h_mask, t_mask, tau: float = 0.5, ce_w: float = 0.1, tv_w: float = 0.3):
    """Best-of-K (per-column) + CE на выбор ветки + TV-консистентность (ветка меняется только на разломах).
    sdf_k [B,K,T,H], logit [B,K,H], true [B,1,T,H].
    ROGII_MTP_ORACLE=1: CE-таргет = мода с нуль-кроссингом ближайшим к GT (оракульный селектор);
    ROGII_MTP_CE_W: переопределяет ce_w (усиление селектора)."""
    import os as _os
    ce_w = float(_os.environ.get("ROGII_MTP_CE_W", ce_w))
    B, K, T, H = sdf_k.shape
    w = (h_mask[:, None, None, :] * t_mask[:, None, :, None]) * torch.exp(-torch.abs(true) / tau)
    se = (sdf_k - true) ** 2
    perc = (w * se).sum(2) / (w.sum(2) + 1e-8)
    if _os.environ.get("ROGII_MTP_ORACLE", "0") == "1":
        with torch.no_grad():
            gt_row = true.squeeze(1).abs().argmin(1)
            pr_row = sdf_k.abs().argmin(2)
            best = (pr_row - gt_row[:, None, :]).abs().argmin(1)
    else:
        best = perc.argmin(1)
    colmask = (h_mask > 0).float()
    sdf_l = (perc.min(1).values * colmask).sum() / (colmask.sum() + 1e-8)
    ce = torch.nn.functional.cross_entropy(logit, best, reduction="none")
    ce = (ce * colmask).sum() / (colmask.sum() + 1e-8)

    prob = torch.softmax(logit, dim=1)
    tv = (prob[:, :, 1:] - prob[:, :, :-1]).abs().sum(1)
    cm2 = colmask[:, 1:] * colmask[:, :-1]
    tv = (tv * cm2).sum() / (cm2.sum() + 1e-8)
    return sdf_l + ce_w * ce + tv_w * tv
