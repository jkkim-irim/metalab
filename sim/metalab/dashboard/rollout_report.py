"""Eval rollout report — ONE HTML per recording: the live 3D scene (left) + time-synced plots (right).

Pairs the ``.rrd`` rerun recording the sim writes (``backends/newton/viewer``) with the series
:class:`sim.metalab.dashboard.rollout_log.PerEnvRolloutLog` records, so a checkpoint can be *inspected* and
*read* at the same instant. The 3D pane is a real renderer, not pixels: orbit, zoom, and look behind the
hand — which is the whole reason it replaced the fixed-camera video pane, since "did the thumb actually wrap
the handle" is not answerable from one angle.

TIME. The rerun viewer owns the clock (its own time panel supplies play/pause/speed/loop) and this page
FOLLOWS it on rAF; clicking a plot writes back through ``set_time_for_timeline``. It works because the sim
stamps the recording on a clock of its own — frame k at exactly k control periods (``_rerun_dt``) — so the
one temporal timeline is simultaneously (a) playable at 1x, where one second of wall clock is one second of
policy time, and (b) an exact index: ``sample = t / control_period``, in the nanoseconds rerun counts in.
Both properties are needed. A SEQUENCE timeline carrying the index was tried and reverted: exact, but it
has no wall-clock meaning, so playback raced the plots ahead of the robot. Raw ``_sim_time`` fails the other
way — the reset settle advances it with rendering off, leaving gaps that break the affine mapping. The
mapping is re-derived from the recording's own range at load time and falls back to fitting if a recording
does not match that cadence.

CHANNELS. Top level = the recorded env (``env0 ✓`` / ``env1 ✗``). Inside it the plot side is a GRID of up to
four panels, and the panel set is exactly the set of channels checked in the picker (grouped by section:
``actor obs`` / ``critic obs`` / ``reward`` / ``action`` / ``custom``) — no separate split control, because the
checks already say how many plots there are. Each panel owns its channel, its y window and its own offscreen
cache; the x window, the playhead and the env are SHARED, so the same sample sits on the same vertical line in
every panel and one recording drives them all. ``custom`` holds everything that is not a raw contract channel —
the log's joint-state measurements plus the views this module synthesizes (:func:`_add_custom_section`).

NOT SELF-CONTAINED. The series are still inlined as JSON, but the 3D half fetches the version-pinned viewer
bundle from rerun's own hosting (:data:`_RERUN_BASE`, ~39 MB, cached immutably) and reads the sibling
``.rrd``. The pane degrades to a message — plots intact — over ``file://`` or when either fetch fails.


Usage (also the wiring point for the eval/train paths)::

    python -m sim.metalab.dashboard.rollout_report <record_dir> [--out FILE] [--title TEXT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

# The CUSTOM section's pairing plan — the same one the live RL tab uses.
from sim.metalab.dashboard import rl_monitor

#: The rerun web viewer, from rerun's own hosting. VERSION-PINNED on purpose: an .rrd stays readable only
#: across adjacent rerun minor versions, so this moves together with the ``rerun-sdk`` pin in
#: ``sim/metalab/setup/<engine>/pyproject.toml``. On skew the viewer says so instead of going blank.
#: The page fetches the .rrd itself and pushes the bytes into the viewer over its log channel, so the
#: viewer never touches the network for the recording.
_RERUN_BASE = "https://app.rerun.io/version/0.35.0"

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<style>
:root{--ground:#eef1f4;--surface:#fff;--surface2:#e6ebef;--ink:#131a21;--soft:#4c5964;--faint:#7a8894;
--line:#d3dae0;--line2:#b7c1c9;--accent:#0f7d8c;--accent-ink:#0a5a66;--signal:#b06a2c;--ok:#2f8f4e;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;
/* Plot panel greys, kept apart from the page surfaces: white face, grey grid, one darker grey for the spine,
   the zero line and the reset markers. On a chart the series colours should be the only colour. */
--plotbg:#fff;--plotgrid:#dfe3e7;--plotaxis:#9aa4ad}
/* ONE light palette, on purpose — no `prefers-color-scheme` branch. The report used to follow the OS into a
   dark theme, which put a white plot panel in a dark page and lit up the room. The 3D pane below stays dark:
   that is rerun's own viewer chrome, not ours to theme. */
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px;
display:flex;flex-direction:column}
.top{padding:11px 16px 0;border-bottom:1px solid var(--line);flex:none}
.tophd{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:.95rem;font-weight:750;margin:0}h1 span{color:var(--accent-ink)}
.sub{font-family:var(--mono);font-size:.72rem;color:var(--soft)}
.etabs{display:flex;gap:2px;margin-top:9px;flex-wrap:wrap}
.etab{font:inherit;font-size:.86rem;padding:8px 15px;border:none;background:none;color:var(--soft);
cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.etab:hover{color:var(--ink)}
.etab.on{color:var(--accent-ink);border-bottom-color:var(--accent);font-weight:700}
.etab b{font-family:var(--mono);font-size:.75rem;margin-left:5px}
.etab b.y{color:var(--ok)}.etab b.n{color:var(--signal)}
.wrap{flex:1;min-height:0;display:flex;gap:14px;padding:12px 16px 14px}
.left{flex:0 1 50%;min-width:22rem;display:flex;flex-direction:column;gap:9px}
.right{flex:1;min-width:0;display:flex;flex-direction:column;gap:9px}
/* Wide-plot toggle: the 3D pane gives up half its width and the plot takes it (50:50 -> 25:75). The min-width
   has to come down with it, or 22rem floors the pane above 25% on anything but a very wide window. */
.wrap.wide .left{flex-basis:25%;min-width:12rem}
#wide{margin-left:auto}
#wide.on{background:var(--accent);border-color:var(--accent);color:#fff}
/* 3D stage: the rerun viewer's canvas fills it absolutely, so the pane keeps its flex height. */
#stage{position:relative;flex:1;min-height:22rem;background:#0d1011;border:1px solid var(--line);
border-radius:8px;overflow:hidden}
#stage canvas{position:absolute;top:0;left:0;width:100%;height:100%;display:block}
#stage canvas.hidden{display:none}
#vstatus{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;
padding:18px;font-size:.78rem;line-height:1.7;color:#8b979e}
#vstatus.hidden{display:none}
#vstatus .err{color:var(--signal)}
#vstatus code{background:#1b2124;border-radius:4px;padding:1px 5px;font-size:.7rem;word-break:break-all}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.lbl{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}
button{font:inherit;font-size:.78rem;padding:5px 11px;border:1px solid var(--line2);border-radius:6px;
background:var(--surface);color:var(--soft);cursor:pointer}
button:hover{color:var(--ink);border-color:var(--accent)}
.seg{display:inline-flex;border:1px solid var(--line2);border-radius:7px;overflow:hidden}
.seg button{border:none;border-right:1px solid var(--line);border-radius:0}
.seg button:last-child{border-right:none}
.seg button.on{background:var(--accent);color:#fff}
.read{font-family:var(--mono);font-size:.72rem;color:var(--soft);line-height:1.6}
.read b{color:var(--accent-ink)}
.ctabs{flex:none;display:flex;gap:7px;flex-wrap:wrap;align-items:center;padding-bottom:2px}
/* Same two-combo channel picker as the Launchpad's RL tab (dashboard/page.py #psec/#pch): section
   above, channel within it below. 48 channels as buttons wrapped into four rows and ate the plot's height. */
.csel{font-family:var(--mono);font-size:.74rem;padding:5px 9px;border:1px solid var(--line2);
border-radius:7px;background:var(--surface);color:var(--ink);cursor:pointer;max-width:100%}
.csel:focus{outline:2px solid var(--accent);outline-offset:1px}
/* Channel multi-picker: a button plus an absolutely-positioned checkbox list. Native <select multiple> is not
   an option — no checkboxes, ctrl-click semantics, and it renders as a box that eats the plot's height. */
.chwrap{position:relative;display:inline-flex}
#chbtn b{color:var(--accent-ink)}
.chpop{position:absolute;top:calc(100% + 4px);left:0;z-index:20;width:20rem;max-height:60vh;overflow:auto;
background:var(--surface);border:1px solid var(--line2);border-radius:8px;padding:6px;
box-shadow:0 8px 24px rgba(0,0,0,.18)}
.chpop[hidden]{display:none}
.chpop .sec{position:sticky;top:-6px;background:var(--surface);font-size:.68rem;text-transform:uppercase;
letter-spacing:.05em;color:var(--faint);padding:5px 6px 3px}
.chpop label{display:flex;align-items:center;gap:7px;padding:3px 6px;border-radius:5px;
font-family:var(--mono);font-size:.72rem;cursor:pointer}
.chpop label:hover{background:var(--surface2)}
.chpop label.dis{opacity:.42;cursor:not-allowed}
.chpop .cap{padding:5px 6px;font-size:.68rem;color:var(--signal)}
.card{flex:1;min-height:0;display:flex;flex-direction:column;background:var(--surface);
border:1px solid var(--line);border-radius:8px;overflow:hidden}
/* Panel grid: 1 / 1x2 / 2x2, driven by --pc/--pr (buildPanels sets them). `minmax(0,1fr)` and min-*:0 on the
   panel are what let a grid child SHRINK — without them a wide canvas ratchets the whole column open. */
.pgrid{flex:1;min-height:0;display:grid;gap:9px;
grid-template-columns:repeat(var(--pc,1),minmax(0,1fr));grid-template-rows:repeat(var(--pr,1),minmax(0,1fr))}
.pane{min-width:0;min-height:0}
/* Focus ring: the top combos, all/none and the series strip all act on THIS panel. */
.pane.on{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.cardhd{flex:none;display:flex;align-items:baseline;gap:9px;padding:7px 10px 0}
.cardhd .t{font-family:var(--mono);font-size:.74rem;font-weight:700;color:var(--accent-ink)}
.cardhd .u{font-size:.68rem;color:var(--faint)}
/* Series checkboxes on ONE line above the chart, scrolled sideways. Vertically they were a 11.5rem column
   that cost a quarter of the card on a desktop and half the row on a phone; horizontally they cost one line.
   Values are deliberately NOT here — they show at the pointer while paused (see drawTip). */
.serstrip{flex:none;display:flex;gap:11px;align-items:center;overflow-x:auto;overflow-y:hidden;
white-space:nowrap;padding:3px 10px 4px;font-family:var(--mono);font-size:.7rem}
.serstrip .lg{display:inline-flex;align-items:center;gap:5px;flex:none;cursor:pointer}
.serstrip .lg.off{opacity:.4}
.serstrip .lg input{margin:0;cursor:pointer}
.serstrip .sw{width:9px;height:9px;border-radius:2px;flex:none}
.plotrow{flex:1;min-height:0;display:flex;padding:4px 10px 9px}
/* val/SR is a counter table, not a series: it replaces the chart inside the same card (see renderTable). The
   `hidden` attribute needs saying twice — an author `display` beats the UA's `[hidden]` rule whatever the
   specificity (cascade ORIGIN wins), so the swapped-out chart kept its box. */
.plotrow[hidden],.serstrip[hidden],.srwrap[hidden]{display:none}
.srwrap{flex:1;min-height:0;overflow:auto;padding:8px 10px 10px}
.srt{border-collapse:collapse;font-family:var(--mono);font-size:.74rem}
.srt th,.srt td{border:1px solid var(--line);padding:4px 11px;text-align:right;white-space:nowrap}
.srt thead th,.srt tbody th{text-align:left;color:var(--faint);font-weight:600}
.srt b{color:var(--accent-ink)}
.srt tr.tot th,.srt tr.tot td{border-top:2px solid var(--line2)}
.chartwrap{flex:1;min-width:0;position:relative}
#cv{cursor:ew-resize;touch-action:none}          /* press-and-drag scrubs the playhead */
canvas{width:100%;height:100%;display:block;cursor:crosshair}
/* Phone: stack the two panes. The 3D one keeps its full height — rerun warns "Mobile OSes are not yet
   supported" but still renders, and it is the thing worth looking at. The plot needs nothing else either:
   its legend is drawn INSIDE the canvas, so the chart owns the full width at every size. */
@media(max-width:900px){.wrap{flex-direction:column}.left{flex:none;max-width:none}.right{min-height:70vh}
.plotrow{padding:6px 8px 8px}
/* Stacked panes have no width to trade, so the toggle is off the table here — and its rule out-specifies
   `.left` above, so neutralise it explicitly rather than relying on order. */
.wrap.wide .left{flex:none;min-width:0}#wide{display:none}}
</style></head><body>
<div class="top">
  <div class="tophd">
    <h1>MetaLab · <span>eval rollout</span></h1>
    <div class="sub" id="meta"></div>
  </div>
  <div class="etabs" id="etabs"></div>
</div>
<div class="wrap" id="wrap">
  <div class="left">
    <div id="stage"><canvas id="the_canvas_id"></canvas><div id="vstatus">loading 3D viewer…</div></div>
    <div class="bar">
      <button id="play" title="play / pause (space)">⏸ pause</button>
      <button id="prev" title="one policy step back (←)">◀ step</button>
      <button id="next" title="one policy step forward (→)">step ▶</button>
    </div>
    <div class="read" id="readout"></div>
  </div>
  <div class="right">
    <div class="ctabs" id="ctabs">
      <span class="chwrap">
        <button id="chbtn" class="csel" title="pick the channels to show — one panel per check, up to 4">
          channels <b id="chn">1</b>/4 ▾</button>
        <div class="chpop" id="chpop" hidden></div>
      </span>
      <span id="lgctl"><button id="lall">all</button><button id="lnone">none</button>
      <span class="lbl" id="sercnt"></span></span>
      <button id="wide" title="give the plot the room: the 3D pane drops to half its width (50:50 → 25:75)">
        ⤢ wide</button></div>
    <div class="serstrip" id="sers"></div>
    <div class="pgrid" id="pgrid"></div>
  </div>
</div>
<script>
const R=__PAYLOAD__, M=R.meta, D=R.data;
const PALETTE=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#00b8b8","#f032e6","#9a6324",
  "#469990","#808000","#e6beff","#fabed4","#42d4f4","#ffd8b1","#000075","#a9a9a9"];
const col=i=>PALETTE[i%PALETTE.length];
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim()||"#888";
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const short=j=>String(j).replace(/_Joint$/,"");
const $=id=>document.getElementById(id);
const clamp=(v,a,b)=>v<a?a:(v>b?b:v);
// A LIVE sub-milli signal must never read as "0.000" — that is indistinguishable from a term that is
// genuinely off, and reward terms land there routinely: the driver pays weight*value*dt, and dt = 0.01 s
// pushes a bounded shaping term (value<=1, weight 0.5) to ~5e-4 per step. The series itself carries 4
// significant digits (rollout_log._num), so this was purely the readout throwing them away.
function fmt(x,d){
  if(typeof x!=="number"||!isFinite(x)) return "—";
  if(d!==undefined) return x.toFixed(d);
  const a=Math.abs(x);
  return (a!==0 && a<5e-4) ? x.toExponential(2) : x.toFixed(3);
}

// --- sync contract (see dashboard/rollout_log.py): sample index <-> wall time -----------------------
// One .rrd frame and one series row per POLICY step (backend._emit_viewer_frames and
// PerEnvRolloutLog.after_step both run once per step), so index <-> index is 1:1 and the only
// conversion left is for display.
const FPS=M.fps, STEP=1;
const timeAt=i=>i/FPS;                                 // series index -> seconds into the rollout

let ENV=String(M.envs[0].id), IDX=0;                   // the recording and the clock are shared by every panel
const VIS={};                                          // tab key -> Set of visible dims (kept across envs)
const tabOf=k=>M.tabs.find(t=>t.key===k);
const nSamples=e=>D[e].step.length;
function vis(k){ if(!VIS[k]) VIS[k]=new Set(tabOf(k).labels.map((_,i)=>i)); return VIS[k]; }
const isTable=t=>!!(t&&t.kind);                        // val/SR: a counter to read, not a series to plot
// Per-series pen. A `rows` tab (the CUSTOM overlays) groups its series per joint, two to a row: give the JOINT
// one colour and tell the pair apart by FILL — a ring for the first source (tgt), a solid dot for the second
// (pos). 22 separate palette colours would both wrap the palette and hide the pairing, which is the view's
// whole point. Cached on the tab: the palette never changes.
function styleOf(t){
  if(t._sty) return t._sty;
  const s=t.labels.map((_,i)=>({c:col(i),fill:true}));
  (t.rows||[]).forEach((r,j)=>r.items.forEach((it,k)=>{ s[it[0]]={c:col(j),fill:k>0}; }));
  return (t._sty=s);
}

// --- header + tabs -------------------------------------------------------------------------------
$("meta").textContent=`${M.task} · ${M.engine} · ${M.hz} Hz policy · ${M.envs.length} env`
  +` · ${nSamples(ENV)} samples · ${(nSamples(ENV)/FPS).toFixed(1)} s`;
$("etabs").innerHTML=M.envs.map(e=>`<button class="etab" data-e="${e.id}">env${e.id}`
  +`<b class="${e.success?"y":"n"}">${e.success?"✓":"✗"}</b></button>`).join("");
// --- the 3D viewer: rerun, embedded, driving the playhead ----------------------------------------
// The VIEWER owns the clock (its own time panel gives play/pause/speed/loop for free) and this page
// follows it on rAF; seeking from a plot writes back. PLAYER is the whole surface the rest of the file
// touches, and stays a null-object when there is no recording — the plots keep working either way.
// The VIEWER plays; this page READS the cursor it is currently rendering. That direction is chosen because it
// is the only one where the plots cannot disagree with the picture: whatever frame is on screen is the frame
// the cursor reports, at any render rate. Driving the cursor from here was tried twice and fails the other
// way — writing ~60 cursor commands a second outruns a heavy scene's render loop, so the robot falls behind
// while the plot cursor keeps perfect time, which reads as "playback is too slow".
//
// Rate is rerun's to get right and it does: it advances the timeline by `stable_dt * speed` per frame, so a
// slow render drops frames instead of stretching the duration. The meter below states the measured number
// rather than asking anyone to trust that.
const TL="step";                        // the sim stamps this with the policy-step index
const PLAYER={ready:false,h:null,rec:null,
  // Non-finite happens for real: the getter returns undefined until the viewer has placed a cursor on the
  // timeline. Math.round(undefined) is NaN, and clamp() passes NaN through, so one bad read used to freeze
  // the whole readout at "sample NaN" forever. Hold the last good index instead.
  idx(){ const t=this.h.get_time_for_timeline(this.rec,TL);
         return Number.isFinite(t)?Math.round(t):IDX; },
  // A seek lands on the viewer's next frame, so for a moment it still reports the OLD time. Without this
  // window the follow loop reads that and yanks the cursor back to where the click came from.
  quiet:0,
  seek(i){ if(this.ready){ this.h.set_time_for_timeline(this.rec,TL,i);
                           this.quiet=performance.now()+250; } },
  following(){ return this.ready && performance.now()>=this.quiet; },
  playing(){ return this.ready && this.h.get_playing(this.rec); },
  toggle(){ if(this.ready) this.h.set_playing(this.rec,!this.h.get_playing(this.rec)); }};

// Playback-rate meter: simulated seconds covered per wall second, measured. 1.00x means one second of robot
// time per second of yours — and with 60 Hz policy steps, one step per 1/60 s.
const RATE={wall0:0,idx0:0,on:false,val:NaN};
function rateMeter(){
  if(!PLAYER.playing()||!PLAYER.following()){ RATE.on=false; RATE.val=NaN; return; }
  const now=performance.now();
  if(!RATE.on){ RATE.on=true; RATE.wall0=now; RATE.idx0=IDX; return; }
  const dw=(now-RATE.wall0)/1000;
  if(dw>=0.75){ RATE.val=((IDX-RATE.idx0)/FPS)/dw; RATE.wall0=now; RATE.idx0=IDX; }
}

const vstat=(html,err)=>{ const s=$("vstatus");
  s.innerHTML=err?`<div><p class="err"><b>${html}</b></p><p class="hint">${err}</p></div>`:html;
  s.classList.remove("hidden"); };

async function bootViewer(){
  // Absolute, because the version-pinned viewer bundle lives at a shared prefix, not beside this report.
  const BASE="__RERUN_BASE__";
  await new Promise((res,rej)=>{ const s=document.createElement("script");
    s.src=BASE+"/re_viewer.js"; s.onload=res; s.onerror=()=>rej(new Error("re_viewer.js"));
    document.head.appendChild(s); });
  const wres=await fetch(BASE+"/re_viewer_bg.wasm");
  if(!wres.ok) throw new Error("re_viewer_bg.wasm: "+wres.status+" "+wres.statusText);
  await wasm_bindgen(wres);

  // EVERY panel minimised: the pane is small, and the time panel was counting rerun's automatic `log_tick`
  // (one tick per log CALL — 51 161 of them for a 600-step rollout) which reads as a bogus step count.
  // Capitalised variants on purpose: the npm wrapper takes "expanded", the raw wasm binding this page talks
  // to deserialises the Rust enum and rejects the lowercase form.
  const h=new wasm_bindgen.WebHandle({url:[],hide_welcome_screen:true,enable_history:false,persist:false,
    panel_state_overrides:{time:"Hidden",blueprint:"Hidden",selection:"Hidden",top:"Hidden"}});
  await h.start($("the_canvas_id"));

  // OUR fetch, then the bytes go in over the viewer's log channel: rerun's own loader omits credentials
  // (ehttp defaults to Credentials::Omit), so it cannot read a recording behind any auth. Fetching it
  // ourselves also keeps the viewer off the network entirely.
  vstat("fetching recording…");
  const res=await fetch(new URL(M.rrd,location.href).href,{credentials:"same-origin"});
  if(!res.ok) throw new Error(M.rrd+": "+res.status+" "+res.statusText);
  const bytes=new Uint8Array(await res.arrayBuffer());
  vstat(`decoding ${(bytes.length/1e6).toFixed(1)} MB…`);
  h.open_channel("metalab","metalab-rollout");
  h.send_rrd_to_channel("metalab",bytes);

  // Decoding is async inside the viewer, so wait for the recording AND for the step range to STOP GROWING:
  // on a partially decoded recording the play-state rewind below would land mid-rollout.
  const n=nSamples(String(M.envs[0].id));
  let prev=-1;
  for(let k=0;k<400;k++){
    const rec=h.get_active_recording_id();
    const r=rec&&h.get_timeline_time_range(rec,TL);
    const settled=r&&r.max>r.min&&r.max===prev;
    prev=r?r.max:-1;
    if(settled){
      // PIN THE AXIS. A recording also carries rerun's automatic `log_time` and `log_tick`, and log_tick
      // counts one tick per log CALL — 51 161 of them for this 600-step rollout. Whichever timeline is
      // active is what the viewer renders AND what its panel counts, so keying the plots to one axis while
      // the picture ran on another is what kept them out of step.
      h.set_active_timeline(rec,TL);
      // No conversion at all: the value IS the sample index (the sim stamps it). The recording's own
      // blueprint pins this timeline's fps to the policy rate, so one step of playback is 1/fps wall clock.
      if(r.max!==n-1)
        console.warn(`rrd step range ends at ${r.max}, series has ${n-1} samples — mapping is off`);
      // rerun's TimeControl defaults to `playing: true, following: true` (Following), which pins the cursor
      // to the NEWEST data — scene frozen on the last frame, get_playing() false because Following is not
      // Playing. Playing clears `following` and rewinds when parked at the end.
      Object.assign(PLAYER,{h,rec});
      h.set_playing(rec,true);
      // `ready` gates the follow loop, so flip it only after the state change lands — otherwise one read
      // happens while the cursor is still parked at the end and the plot cursor starts life pinned there.
      await new Promise(r2=>setTimeout(r2,200));
      PLAYER.ready=true;
      return;
    }
    await new Promise(r2=>setTimeout(r2,50));
  }
  throw new Error("the recording never exposed a settled '"+TL+"' timeline");
}

if(!M.rrd){
  vstat("This recording has no .rrd (3D needs the newton spoke with --rrd).");
}else if(!location.protocol.startsWith("http")){
  vstat("3D needs the report served over http(s)",
        "Opened from a local file, where the viewer bundle and the recording cannot be fetched.");
}else{
  bootViewer().then(()=>{
    $("vstatus").classList.add("hidden");
    setPlayLabel();
  }).catch(e=>vstat("3D viewer failed to load",
    String(e&&e.message||e)+" — the plots below still work, and the sibling "
    +"<code>rollout.rrd</code> opens in a local rerun viewer."));
}

const SECTS=[]; M.tabs.forEach(t=>{ if(!SECTS.some(s=>s.name===t.section)) SECTS.push({name:t.section,tabs:[]});
  SECTS.find(s=>s.name===t.section).tabs.push(t); });
// The CHECKED SET IS THE PANEL SET: no split control, no per-panel picker. Panels are drawn in list order, so
// the grid is a pure function of the checks and re-checking never shuffles what you were looking at.
const MAXCH=4;                                    // 2x2 is the ceiling; at 4 the unchecked boxes go disabled
const CHK=new Set([M.tabs[0].key]);
function chPop(){
  $("chpop").innerHTML=SECTS.map(s=>`<div class="sec">${esc(s.name)}</div>`
    +s.tabs.map(t=>{const on=CHK.has(t.key), dis=!on&&CHK.size>=MAXCH;
      return `<label class="${dis?"dis":""}" title="${esc(t.title)}${t.unit?" ["+esc(t.unit)+"]":""}">`
        +`<input type="checkbox" data-k="${esc(t.key)}"${on?" checked":""}${dis?" disabled":""}>`
        +`${esc(t.title)}</label>`;}).join("")).join("")
    +(CHK.size>=MAXCH?`<div class="cap">4 panels max — uncheck one to pick another</div>`:"");
  $("chn").textContent=CHK.size;
}
$("chbtn").onclick=e=>{ e.stopPropagation(); const pop=$("chpop"); pop.hidden=!pop.hidden; if(!pop.hidden) chPop(); };
$("chpop").addEventListener("click",e=>e.stopPropagation());
$("chpop").addEventListener("change",e=>{
  const cb=e.target.closest("input[type=checkbox]"); if(!cb) return;
  const k=cb.dataset.k;
  if(cb.checked){ if(CHK.size>=MAXCH){cb.checked=false; return;} CHK.add(k); }
  else if(CHK.size===1){ cb.checked=true; return; }             // the last panel cannot be closed
  else CHK.delete(k);
  chPop();
  buildPanels(M.tabs.filter(t=>CHK.has(t.key)).map(t=>t.key));   // list order, not click order
  drawAll(); readout();
});
document.addEventListener("click",()=>{ $("chpop").hidden=true; });
window.addEventListener("keydown",e=>{ if(e.key==="Escape") $("chpop").hidden=true; });
function setPlayLabel(){ $("play").textContent=PLAYER.playing()?"⏸ pause":"▶ play"; }

// --- panels -------------------------------------------------------------------------------------
// One panel = one plot. Everything the chart layer needs per plot lives on the panel object; what is genuinely
// shared stays module-level:
//   * ENV / IDX — one recording, one clock. A playhead per panel would break the 3D sync contract.
//   * VIEW.i0/i1 — the x window is LINKED across panels, which is the whole point of comparing side by side:
//     the same sample sits on the same vertical line in every panel. y is per panel (units differ).
//   * VIS[tab key] — the series picker is keyed by CHANNEL, so two panels on one channel share their pick.
// The FOCUSED panel is what the top combos, all/none and the series strip edit; clicking a panel focuses it.
const PANELS=[];      // [{tab, el, hdt, hdu, cv, off, plotrow, srt, sig, geo, y, tip, srsig}]
let FOCUS=0;
const pfocus=()=>PANELS[FOCUS];
function geom(w,h){ return {L:56,R:10,T:10,B:20,pw:w-66,ph:h-30,w:w,h:h}; }

// The x window, in sample indices, SHARED (see above). Per-panel y lives in `p.y`: a pinned [min,max] or null
// = auto-fit, which renderStatic refits over the samples in view.
let VIEW={i0:0,i1:0};
function viewFit(n){ VIEW.i0=0; VIEW.i1=Math.max(1,n-1); }
// Repair, not policy: a window from a longer env (or the pre-first-render zero one) cannot address this
// recording, so it falls back to the whole of it. Deliberate resets live in selectEnv/setPanelTab.
function viewClamp(n){ if(!(VIEW.i1>VIEW.i0)||VIEW.i0<0||VIEW.i1>Math.max(1,n-1)) viewFit(n); }

// Zoom about a pointer position, in window units (the event wiring is with the other interactions below). x is
// the shared window — one panel's wheel moves every panel's time axis; y is the panel's own.
const MIN_SPAN=2;                               // 3 samples in view: past that the polyline is one segment
function zoomX(p,px,s){
  const {G,kAt,n,span}=p.geo, ka=kAt(px), f=(px-G.L)/G.pw;
  const sp=clamp(span/s,MIN_SPAN,Math.max(1,n-1));
  let i0=ka-f*sp, i1=i0+sp;                     // the sample under the pointer stays under it
  if(i0<0){ i1-=i0; i0=0; }                     // slid back inside rather than clipped: clipping would eat
  if(i1>n-1){ i0-=i1-(n-1); i1=n-1; }           // the span again and stall the zoom at either edge
  VIEW.i0=Math.max(0,i0); VIEW.i1=Math.min(Math.max(1,n-1),i1);
}
function zoomY(p,py,s){
  const {G,vAt,ymin,ymax}=p.geo, va=vAt(py), f=(py-G.T)/G.ph, sp=(ymax-ymin)/s;
  if(!(sp>0)||!isFinite(sp)) return;
  p.y=[va-(1-f)*sp, va+f*sp];                   // PINS y — auto-fit resumes on shift+wheel or dblclick
}

function renderStatic(p){
  const cv=p.cv, dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth, h=cv.clientHeight;
  if(!w||!h) return false;
  const V=vis(p.tab), n=nSamples(ENV);
  viewClamp(n);
  const i0=VIEW.i0, i1=VIEW.i1, span=i1-i0;
  // The zoom window is part of the cache key: without it a zoom would silently re-blit the old bitmap.
  const sig=[ENV,p.tab,w,h,dpr,[...V].join(","),i0,i1,p.y?p.y.join(","):"auto"].join("|");
  if(sig===p.sig) return true;
  p.sig=sig;
  cv.width=p.off.width=Math.round(w*dpr); cv.height=p.off.height=Math.round(h*dpr);
  const g=p.off.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  g.font="10px "+cssv("--mono");
  const G=geom(w,h), env=D[ENV], series=env.series[p.tab];
  // Y range over the VISIBLE dims: the window the user pinned, else auto-fit over the samples IN VIEW, so
  // zooming x magnifies a flat stretch. Refit on the WINDOW only — the axis holds still under a moving cursor.
  let ymin,ymax;
  if(p.y){ ymin=p.y[0]; ymax=p.y[1]; }
  else{
    ymin=Infinity;ymax=-Infinity;
    const k0=Math.max(0,Math.floor(i0)), k1=Math.min(n-1,Math.ceil(i1));
    V.forEach(i=>{const a=series[i]||[];for(let k=k0;k<=k1&&k<a.length;k++){const v=a[k];
      if(v!==null&&isFinite(v)){if(v<ymin)ymin=v;if(v>ymax)ymax=v;}}});
    if(!isFinite(ymin)){ymin=-1;ymax=1;} if(ymin===ymax){ymin-=1;ymax+=1;}
    const pd=(ymax-ymin)*0.08; ymin-=pd; ymax+=pd;
  }
  const X=k=>G.L+G.pw*(span>0?(k-i0)/span:0), Y=v=>G.T+G.ph*(1-(v-ymin)/(ymax-ymin));
  // Inverses, in the window's units: what the pointer is over. Both directions are needed to anchor a zoom.
  const kAt=px=>i0+(span>0?span*(px-G.L)/G.pw:0), vAt=py=>ymax-(ymax-ymin)*(py-G.T)/G.ph;
  p.geo={G:G,X:X,Y:Y,kAt:kAt,vAt:vAt,ymin:ymin,ymax:ymax,n:n,i0:i0,i1:i1,span:span};
  // Ticks first: the grid lines and their labels have to land on the same values.
  const ys=[]; for(let k=0;k<=4;k++) ys.push(ymax-(ymax-ymin)*k/4);
  const nx=Math.max(2,Math.min(5,Math.floor(G.pw/72)));   // ~72 px per label, else they overprint
  const xs=[];                                            // zoomed in past one sample per label: integer ticks
  if(span<=nx) for(let k=Math.ceil(i0);k<=Math.floor(i1);k++) xs.push(k);
  else for(let k=0;k<=nx;k++) xs.push(Math.round(i0+span*k/nx));
  // The panel: white face, grey grid both ways, and a spine so it reads as a panel and not a hole. Lines are
  // snapped to the half pixel — at dpr 1 an integer coordinate straddles two rows and greys out.
  g.fillStyle=cssv("--plotbg"); g.fillRect(G.L,G.T,G.pw,G.ph);
  g.strokeStyle=cssv("--plotgrid"); g.lineWidth=1; g.beginPath();
  ys.forEach(v=>{const yy=Math.round(Y(v))+.5; g.moveTo(G.L,yy); g.lineTo(G.L+G.pw,yy);});
  xs.forEach(k=>{const xx=Math.round(X(k))+.5; g.moveTo(xx,G.T); g.lineTo(xx,G.T+G.ph);});
  g.stroke();
  g.strokeStyle=cssv("--plotaxis"); g.strokeRect(G.L+.5,G.T+.5,G.pw-1,G.ph-1);
  g.fillStyle=cssv("--faint");                            // labels are outside the panel: page-themed
  g.textAlign="right";g.textBaseline="middle";
  ys.forEach(v=>g.fillText(fmt(v,2),G.L-6,Y(v)));
  g.textAlign="center";g.textBaseline="top";
  xs.forEach(kk=>g.fillText(kk,X(kk),G.T+G.ph+4));
  // Everything in DATA coordinates is clipped to the plot box from here on: zoomed or panned, a trace that
  // runs off the window must not paint over the y labels or outside the card.
  g.save(); g.beginPath(); g.rect(G.L,G.T,G.pw,G.ph); g.clip();
  if(ymin<0&&ymax>0){const zy=Math.round(Y(0))+.5; g.strokeStyle=cssv("--plotaxis");
    g.beginPath();g.moveTo(G.L,zy);g.lineTo(G.L+G.pw,zy);g.stroke();}
  // episode boundaries: the recorded window spans resets — success resets are marked apart from the rest.
  // Panel-fixed greys, not the page's --line2: on the white face the light theme's would vanish into the grid.
  g.setLineDash([3,3]);
  for(let k=0;k<n;k++){ if(!env.done[k]||k<i0||k>i1) continue;
    g.strokeStyle=env.success[k]?cssv("--ok"):cssv("--plotaxis"); g.globalAlpha=.85;
    g.beginPath();g.moveTo(X(k),G.T);g.lineTo(X(k),G.T+G.ph);g.stroke(); }
  g.setLineDash([]); g.globalAlpha=1;
  // `line:false` = SCATTER only (the target-vs-pos views): a line through the samples interpolates motion that
  // never happened, and a transport delay is exactly N discrete flat samples. Markers are drawn only for the
  // samples IN VIEW — at 600 samples × 22 series a full pass runs on every wheel notch.
  const t=tabOf(p.tab), pen=styleOf(t), drawLine=t.line!==false, drawDots=!!t.marker||!drawLine;
  const d0=Math.max(0,Math.floor(i0)), d1=Math.min(n-1,Math.ceil(i1));
  V.forEach(i=>{const a=series[i]||[]; if(!a.length)return;
    const st=pen[i]||{c:col(i),fill:true};
    if(drawLine){
      g.strokeStyle=st.c; g.lineWidth=1.3; g.beginPath();
      let on=false;                                 // lift the pen over null (a gap, not a spike to zero)
      for(let k=0;k<a.length;k++){const v=a[k];
        if(v===null||!isFinite(v)){on=false;continue;}
        const px=X(k),py=Y(v); on?g.lineTo(px,py):g.moveTo(px,py); on=true;}
      g.stroke();
    }
    if(drawDots){
      g.strokeStyle=st.c; g.fillStyle=st.c; g.lineWidth=1.2;
      for(let k=d0;k<=d1;k++){const v=a[k];
        if(v===null||v===undefined||!isFinite(v))continue;
        g.beginPath(); g.arc(X(k),Y(v),st.fill?2:2.7,0,7); st.fill?g.fill():g.stroke();}
    }});
  g.restore();
  return true;
}

// --- dynamic layer: blit the static plot + the playhead cursor ------------------------------------
const drawAll=()=>PANELS.forEach(drawFrame);
function drawFrame(p){
  if(isTable(tabOf(p.tab))){ renderTable(p); return; }   // this panel holds a table, not a canvas
  if(!renderStatic(p)||!p.geo) return;
  const cv=p.cv, dpr=window.devicePixelRatio||1, g=cv.getContext("2d");
  g.setTransform(1,0,0,1,0,0); g.clearRect(0,0,cv.width,cv.height); g.drawImage(p.off,0,0);
  g.setTransform(dpr,0,0,dpr,0,0);
  const {G,X,Y}=p.geo, series=D[ENV].series[p.tab], cx=X(IDX);
  // Clipped like the static layer: a playhead outside the zoom window is simply not drawn, and a value dot
  // never lands on the axis gutter. The tooltip is clamped inside the box already, so it stays outside.
  g.save(); g.beginPath(); g.rect(G.L,G.T,G.pw,G.ph); g.clip();
  g.strokeStyle=cssv("--accent"); g.globalAlpha=.75; g.lineWidth=1;
  g.beginPath();g.moveTo(cx,G.T);g.lineTo(cx,G.T+G.ph);g.stroke(); g.globalAlpha=1;
  const pen=styleOf(tabOf(p.tab));
  vis(p.tab).forEach(i=>{const v=(series[i]||[])[IDX];
    if(v===null||!isFinite(v))return; g.fillStyle=(pen[i]||{}).c||col(i);
    g.beginPath();g.arc(cx,Y(v),2.6,0,7);g.fill();});
  g.restore();
  drawTip(p,g,G,series);
}

// Value tooltip: only while the pointer is HELD on a paused plot, drawn beside it so the finger or cursor
// does not cover the numbers. Values are not on the checkbox chips on purpose — those are for picking series,
// this is for reading one instant.
const TIP_MAX=14;                               // p.tip = {cx,cy} in that panel's canvas CSS px, or null
function drawTip(p,g,G,series){
  const TIP=p.tip;
  if(!TIP) return;
  const t=tabOf(p.tab), idx=[...vis(p.tab)].sort((a,b)=>a-b);
  if(!idx.length) return;
  const shown=idx.slice(0,TIP_MAX), more=idx.length-shown.length;
  const fs=11, pad=6, sw=8, gap=6, lh=fs+3;
  g.font=`${fs}px ${cssv("--mono")||"monospace"}`; g.textBaseline="middle"; g.textAlign="left";
  const rows=shown.map(i=>({i,txt:`${short(t.labels[i])}  ${fmt((series[i]||[])[IDX])}`}));
  if(more>0) rows.push({i:-1,txt:`+${more} more`});
  const w=Math.max(...rows.map(r=>g.measureText(r.txt).width))+sw+gap+pad*2;
  const h=rows.length*lh+pad*2;
  // right of the pointer, flipped left when it would leave the plot; vertically clamped inside it
  let x=TIP.cx+14; if(x+w>G.L+G.pw) x=TIP.cx-14-w;
  x=clamp(x,G.L,G.L+G.pw-w);
  const y=clamp(TIP.cy-h/2,G.T,G.T+G.ph-h);
  g.globalAlpha=.93; g.fillStyle=cssv("--surface"); g.fillRect(x,y,w,h); g.globalAlpha=1;
  g.strokeStyle=cssv("--accent"); g.lineWidth=1; g.strokeRect(x+.5,y+.5,w-1,h-1);
  const pen=styleOf(t);
  rows.forEach((r,k)=>{
    const cy=y+pad+lh*k+lh/2;
    if(r.i>=0){ g.fillStyle=(pen[r.i]||{}).c||col(r.i); g.fillRect(x+pad,cy-sw/2,sw,sw); }
    g.fillStyle=r.i>=0?cssv("--ink"):cssv("--faint");
    g.fillText(r.txt,x+pad+sw+gap,cy);
  });
}

// The series picker: ONE horizontal line of checkboxes above the chart, scrolled sideways when it overflows.
// No values here on purpose — those are what drawTip shows at the pointer.
// What ONE chip picks. A `rows` tab is picked BY JOINT: its row (tgt + pos) is the thing you look at, so one
// chip carries both series — two chips per joint was just twice the clicking to see one tracking gap. Every
// other tab is one chip per series, as before.
function groups(t){
  return t.rows ? t.rows.map(r=>[r.joint, r.items.map(it=>it[0])])
                : t.labels.map((lab,i)=>[lab,[i]]);
}
// The strip edits the FOCUSED panel's channel — one strip for the grid, not one per panel: 22 checkboxes in a
// quarter-width panel is unreadable, and the pick is per CHANNEL anyway (VIS is keyed by tab).
const serCount=()=>{ const V=vis(pfocus().tab), gs=groups(tabOf(pfocus().tab));
  return `${gs.filter(g=>g[1].some(i=>V.has(i))).length}/${gs.length}`; };
function drawLegend(){
  const k=pfocus().tab, t=tabOf(k), V=vis(k), pen=styleOf(t);
  $("sers").innerHTML=groups(t).map(([lab,ix])=>{
    const st=pen[ix[0]]||{c:col(ix[0]),fill:true}, on=ix.some(i=>V.has(i));
    // A chip is one colour; on a paired tab the ○/● split lives on the plot (and in the card's unit line).
    const sw=st.fill||ix.length>1 ? `background:${st.c}`
                                  : `background:transparent;box-shadow:inset 0 0 0 2px ${st.c}`;
    return `<label class="lg${on?"":" off"}" data-i="${ix.join(",")}" title="${esc(lab)}">`
      +`<input type="checkbox"${on?" checked":""}>`
      +`<span class="sw" style="${sw}"></span>${esc(short(lab))}</label>`;}).join("");
  $("sercnt").textContent=serCount();
}

// val/SR: success/attempts per env, counted over the recorded window UP TO THE PLAYHEAD. The live RL tab counts
// over the whole run from the driver's own counters; a report only has the window it recorded, so this is
// derived from the same `done`/`success` arrays the env tabs' ✓/✗ comes from. Every env at once, like the
// dashboard's table — it ignores the env selector on purpose.
function renderTable(p){                          // p.srsig: counts change at a reset, the playhead at 60/s
  const rows=M.envs.map(e=>{
    const d=D[String(e.id)], upto=Math.min(IDX+1,d.done.length);
    let s=0,a=0;
    for(let k=0;k<upto;k++) if(d.done[k]){ a++; if(d.success[k]) s++; }
    return {id:e.id,s:s,a:a};
  });
  const sig=rows.map(r=>`${r.s}/${r.a}`).join(",");
  if(sig===p.srsig) return;
  p.srsig=sig;
  const pct=(s,a)=>a?`${(100*s/a).toFixed(1)}%`:"—";
  let ts=0,ta=0;
  const body=rows.map(r=>{ts+=r.s;ta+=r.a;
    return `<tr><th>env ${r.id}</th><td><b>${r.s}</b> / ${r.a}</td><td>${pct(r.s,r.a)}</td></tr>`;}).join("");
  p.srt.innerHTML=`<table class="srt"><thead><tr><th>ENV</th><th>SUCCESS / ATTEMPTS</th><th>SR</th></tr>`
    +`</thead><tbody>${body}<tr class="tot"><th>all</th><td><b>${ts}</b> / ${ta}</td>`
    +`<td>${pct(ts,ta)}</td></tr></tbody></table>`
    +`<div class="read" style="margin-top:9px">episodes that ENDED at or before the playhead — one still`
    +` running is not an attempt yet, so this grows as the recording plays</div>`;
}
function readout(){
  const env=D[ENV], n=nSamples(ENV);
  // Two different counters on purpose, so spell out which is which: `sample` indexes the RECORDING (0-based,
  // and it is exactly the .rrd frame the 3D pane is showing), while `ep.step` is the env's own counter inside
  // the CURRENT episode (1-based, restarts at every reset). They differ by design — not a sync error.
  $("readout").innerHTML=`<span title="recording sample = the .rrd frame on the left, 0-based">sample</span>`
    +` <b>${IDX}</b> / ${n-1} &nbsp;·&nbsp; t <b>${timeAt(IDX).toFixed(2)}</b> s`
    +` &nbsp;·&nbsp; <span title="the env's step inside the current episode: 1-based and reset to 1 on every`
    +` episode boundary, so it does not track the sample index">ep.step</span>`
    +` <b>${env.step[IDX]}</b> / ${M.max_step}`
    +(Number.isFinite(RATE.val)
        ? ` &nbsp;·&nbsp; <span title="measured: simulated seconds per wall second. 1.00x = real time">`
          +`playback</span> <b>${RATE.val.toFixed(2)}×</b>` : "")
    +(env.done[IDX]?` &nbsp;·&nbsp; <b>${env.success[IDX]?"SUCCESS":"reset"}</b>`:"");
}

function setIdx(i,{seek=false}={}){
  if(!Number.isFinite(i)) return;              // clamp() would pass NaN straight through and stick
  const n=nSamples(ENV), ni=clamp(i,0,n-1);
  if(seek) PLAYER.seek(ni);
  if(ni===IDX&&!seek){drawAll();return;}
  IDX=ni; drawAll(); readout();
}

// --- interactions --------------------------------------------------------------------------------
// Switching env switches PLOTS only: one recording carries every env at once (the sim tiles them by world
// offset), so there is nothing to show or hide on the 3D side and the playhead is shared.
function selectEnv(e){
  ENV=String(e);
  document.querySelectorAll(".etab").forEach(b=>b.classList.toggle("on",b.dataset.e===ENV));
  // Full repaint, not setIdx(IDX): that early-returns when the index is unchanged, which on the very first
  // call (IDX=0) left the readout and legend values blank until playback moved the playhead.
  IDX=clamp(IDX,0,nSamples(ENV)-1);
  PANELS.forEach(p=>{p.sig="";});
  if(!isTable(tabOf(pfocus().tab))) drawLegend();
  drawAll(); readout();
}
// Point ONE panel (the focused one) at a channel.
function setPanelTab(p,k){
  p.tab=k;
  p.y=null;                                     // a y window pinned in rad means nothing on an N·m channel
  const t=tabOf(k), tbl=isTable(t);
  p.hdt.textContent=t.title; p.hdu.textContent=t.unit;
  // A table channel swaps out the chart inside its own panel; the series strip goes with it when that panel is
  // the focused one — there is nothing to pick, and an "0/2" count next to a table reads like a broken plot.
  p.srt.hidden=!tbl; p.plotrow.hidden=tbl;
  p.sig="";                                     // the canvas was hidden: its cached bitmap is not reusable
  p.srsig="";
}
// The strip is the ONLY thing that still needs a focused panel — channels come from the checkbox list.
function refreshStrip(){
  const tbl=isTable(tabOf(pfocus().tab));
  $("sers").hidden=tbl; $("lgctl").hidden=tbl;
  if(!tbl) drawLegend();
}
// Focus follows a click on a panel (a scrub press counts). Bails when nothing changes — every plot press calls
// this, and it rebuilds the strip's DOM.
function setFocus(i){
  const want=clamp(i,0,PANELS.length-1);
  if(want===FOCUS) return;
  FOCUS=want;
  PANELS.forEach((p,j)=>p.el.classList.toggle("on",j===FOCUS));
  refreshStrip();
}
$("etabs").addEventListener("click",e=>{const b=e.target.closest(".etab"); if(b) selectEnv(b.dataset.e);});
// The pick is stored per CHANNEL, so it repaints every panel showing that channel, not just the focused one.
const repaintTab=k=>PANELS.forEach(p=>{ if(p.tab===k){ p.sig=""; drawFrame(p); } });
$("sers").addEventListener("change",e=>{                    // one checkbox per PICK GROUP (see groups())
  const el=e.target.closest(".lg"); if(!el)return;
  const k=pfocus().tab, V=vis(k), on=e.target.checked;      // the input already flipped: follow it
  el.dataset.i.split(",").forEach(s=>{const i=+s; on?V.add(i):V.delete(i);});
  el.classList.toggle("off",!on);
  $("sercnt").textContent=serCount();
  repaintTab(k);});
$("lall").onclick=()=>{const k=pfocus().tab;
  VIS[k]=new Set(tabOf(k).labels.map((_,i)=>i)); drawLegend(); repaintTab(k);};
$("lnone").onclick=()=>{const k=pfocus().tab; VIS[k]=new Set(); drawLegend(); repaintTab(k);};
// Wide plot: trade width from the 3D pane. A CSS class change fires no resize event, so repaint by hand —
// renderStatic reads clientWidth, which forces the new layout before it measures.
$("wide").onclick=()=>{
  const on=$("wrap").classList.toggle("wide");
  $("wide").classList.toggle("on",on);
  $("wide").textContent=on?"⤡ even":"⤢ wide";
  PANELS.forEach(p=>{p.sig="";}); drawAll();
};

// --- one panel: its DOM and its own pointer/wheel handlers ----------------------------------------
// X gestures (scrub, wheel-x, pan) act on SHARED state, so they repaint every panel; y gestures touch only the
// panel under the pointer. Pressing a panel also focuses it.
let SCRUB=null, PLAYED_BEFORE=false;            // SCRUB = the panel being dragged, or null
let PAN=null;                                   // {p,px,py,i0,i1,y0,y1} captured at the press, or null
const PIN_DY=2;                                 // px of vertical travel before a drag counts as a y pan
// Through the window's own mapping, not the full range: under a zoom, the pixel the pointer is on means a
// different sample, and the 3D pane would seek to the wrong frame.
const posToIdx=(p,e)=>{ const r=p.cv.getBoundingClientRect();
  return Math.round(clamp(p.geo.kAt(e.clientX-r.left),0,p.geo.n-1)); };
const tipAt=(p,e)=>{ const r=p.cv.getBoundingClientRect();
  return {cx:e.clientX-r.left, cy:e.clientY-r.top}; };

function bindPanel(p){
  const cv=p.cv;
  // Scrub: press anywhere on the plot and drag, and the playhead follows the pointer (a click is just a
  // zero-length drag, so this replaces the old click-to-seek). Two details make it behave:
  //   * pointer CAPTURE, so the drag keeps tracking after the cursor leaves the canvas or the window;
  //   * playback PAUSES for the duration — at 1x the clock advances 60 samples a second, which would drag the
  //     cursor out from under the hand whenever it stops moving. The previous state is restored on release.
  cv.addEventListener("pointerdown",e=>{
    setFocus(PANELS.indexOf(p));                            // any press retargets the combos and the strip
    if(!p.geo||e.button!==0)return;
    SCRUB=p; PLAYED_BEFORE=PLAYER.playing();
    if(PLAYED_BEFORE) PLAYER.toggle();                      // hold still while the pointer leads
    else p.tip=tipAt(p,e);                                  // ALREADY paused -> reading values, so show them
    cv.setPointerCapture(e.pointerId);
    setIdx(posToIdx(p,e),{seek:true});
    e.preventDefault();                                     // no text/canvas selection mid-drag
  });
  cv.addEventListener("pointermove",e=>{
    if(SCRUB!==p||!p.geo)return;
    if(p.tip) p.tip=tipAt(p,e);
    setIdx(posToIdx(p,e),{seek:true});
    if(p.tip) drawFrame(p);                                 // same index, moved pointer -> tooltip follows
  });
  const endScrub=e=>{
    if(SCRUB!==p)return;
    SCRUB=null; p.tip=null;
    if(cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId);
    if(PLAYED_BEFORE) PLAYER.toggle();
    setPlayLabel(); drawFrame(p);                           // clear the tooltip
  };
  cv.addEventListener("pointerup",endScrub);
  cv.addEventListener("pointercancel",endScrub);
  // Wheel = zoom anchored on the pointer, so the sample AND the value under it stay put. Plain wheel takes both
  // axes (map-like); `shift` = time only and hands y back to auto-fit, which re-tightens it to whatever is in
  // view; `alt` = value only. NOT passive: without preventDefault the page scrolls instead once the panes stack
  // under 900px, and shift+wheel is a horizontal scroll. `ctrl` is left alone — that is the browser's own zoom.
  cv.addEventListener("wheel",e=>{
    if(!p.geo||e.ctrlKey) return;
    e.preventDefault();
    const r=cv.getBoundingClientRect(), px=e.clientX-r.left, py=e.clientY-r.top, {G}=p.geo;
    if(px<G.L||px>G.L+G.pw||py<G.T||py>G.T+G.ph) return;        // the axis gutters are not the plot
    // deltaY is in lines or pages on some mice and pixels elsewhere; normalise, then cap so one coarse notch
    // cannot jump 100x.
    const d=e.deltaMode===1?e.deltaY*16:(e.deltaMode===2?e.deltaY*100:e.deltaY);
    const s=Math.exp(-clamp(d,-240,240)*0.0018);                // wheel away = deltaY<0 = zoom in
    if(!e.altKey) zoomX(p,px,s);
    if(e.shiftKey) p.y=null; else zoomY(p,py,s);
    if(e.altKey) drawFrame(p); else drawAll();                  // x moved -> every panel's axis moved
  },{passive:false});

  // MIDDLE-button drag = pan, in every direction. Left is taken (scrub), and the middle button is otherwise
  // dead here — the early `e.button!==0` return in the scrub handler above leaves it entirely free.
  cv.addEventListener("pointerdown",e=>{
    if(!p.geo||e.button!==1||SCRUB) return;     // mid-scrub: the window must hold still under the playhead
    PAN={p:p,px:e.clientX,py:e.clientY,i0:p.geo.i0,i1:p.geo.i1,y0:p.geo.ymin,y1:p.geo.ymax};
    cv.setPointerCapture(e.pointerId);
    cv.style.cursor="grabbing";
    e.preventDefault();                         // also kills the compat mousedown = no middle-click autoscroll
  });
  cv.addEventListener("pointermove",e=>{
    if(!PAN||PAN.p!==p||!p.geo) return;
    const {G,n}=p.geo, sp=PAN.i1-PAN.i0, dy=e.clientY-PAN.py;
    let i0=PAN.i0-(e.clientX-PAN.px)*sp/G.pw, i1=i0+sp;   // the grabbed sample follows the pointer
    if(i0<0){ i1-=i0; i0=0; }                             // hits an end: stop, keeping the span (as in zoomX)
    if(i1>n-1){ i0-=i1-(n-1); i1=n-1; }
    VIEW.i0=Math.max(0,i0); VIEW.i1=Math.min(Math.max(1,n-1),i1);
    // A vertical drag PINS y — an axis that is refitting itself cannot be panned. A horizontal-only drag leaves
    // it on auto-fit, so y keeps re-tightening as the window slides. Rebased at the pin, so nothing jumps.
    if(!p.y){
      if(Math.abs(dy)>PIN_DY){ PAN.py=e.clientY; PAN.y0=p.geo.ymin; PAN.y1=p.geo.ymax;
                               p.y=[p.geo.ymin,p.geo.ymax]; }
    }else{
      const dv=(e.clientY-PAN.py)*(PAN.y1-PAN.y0)/G.ph;   // drag down = the trace comes down with the pointer
      p.y=[PAN.y0+dv,PAN.y1+dv];
    }
    drawAll();                                            // the pan moved x: every panel follows
  });
  const endPan=e=>{
    if(!PAN||PAN.p!==p) return;
    PAN=null; cv.style.cursor="";
    if(cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId);
  };
  cv.addEventListener("pointerup",endPan);
  cv.addEventListener("pointercancel",endPan);
  cv.addEventListener("auxclick",e=>e.preventDefault());   // middle click: no autoscroll, no X11 paste
  // Back to the whole recording: x for everyone (it is shared), y for this panel. The left-button pair also
  // scrubs, which is harmless — the playhead lands where it was clicked and the view resets around it.
  cv.addEventListener("dblclick",()=>{ viewFit(nSamples(ENV)); p.y=null; drawAll(); });
  p.srt.addEventListener("pointerdown",()=>setFocus(PANELS.indexOf(p)));
}

// (Re)build the grid from a list of channel keys — one panel per key, in order.
function buildPanels(keys){
  const host=$("pgrid");
  PANELS.length=0; host.innerHTML="";
  keys.forEach(k=>{
    const el=document.createElement("div");
    el.className="card pane";
    el.innerHTML='<div class="cardhd"><span class="t"></span><span class="u"></span></div>'
      +'<div class="plotrow"><div class="chartwrap"><canvas></canvas></div></div>'
      +'<div class="srwrap" hidden></div>';
    const p={tab:k, el:el, hdt:el.querySelector(".t"), hdu:el.querySelector(".u"),
             cv:el.querySelector("canvas"), plotrow:el.querySelector(".plotrow"),
             srt:el.querySelector(".srwrap"), off:document.createElement("canvas"),
             sig:"", geo:null, y:null, tip:null, srsig:""};
    PANELS.push(p); host.appendChild(el); bindPanel(p);
    setPanelTab(p,k);
  });
  host.style.setProperty("--pc",keys.length>1?2:1);        // 1 / 1x2 / 2x2
  host.style.setProperty("--pr",keys.length>2?2:1);
  // The panel ELEMENTS are new, so the focus ring has to be re-applied even at the same index: clear it first
  // so setFocus cannot early-return.
  const want=Math.min(Math.max(FOCUS,0),keys.length-1);
  FOCUS=-1; setFocus(want);
}
$("prev").onclick=()=>setIdx(IDX-STEP,{seek:true});
$("next").onclick=()=>setIdx(IDX+STEP,{seek:true});
$("play").onclick=()=>{PLAYER.toggle(); setPlayLabel();};
window.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft"){setIdx(IDX-STEP,{seek:true});e.preventDefault();}
  else if(e.key==="ArrowRight"){setIdx(IDX+STEP,{seek:true});e.preventDefault();}
  else if(e.key===" "){PLAYER.toggle();setPlayLabel();e.preventDefault();}});
window.addEventListener("resize",()=>{PANELS.forEach(p=>{p.sig="";});drawAll();});

// Polled on rAF: the viewer's cursor moves for reasons this page never hears about (its own play button,
// dragging its scrub bar, loop wrap-around), so reading it is the only direction that cannot go stale.
let WASPLAYING=false;
(function tick(){
  if(PLAYER.following()) setIdx(PLAYER.idx());
  if(PLAYER.ready){
    rateMeter();
    const p=PLAYER.playing();
    if(p!==WASPLAYING){WASPLAYING=p; setPlayLabel(); readout();}
  }
  requestAnimationFrame(tick); })();

buildPanels([M.tabs[0].key]); selectEnv(M.envs[0].id);
</script></body></html>"""


def _state_channels_under_custom(payload: dict) -> None:
    """Re-file a recording's joint-state channels under ``custom``.

    The log tags them that way now; a recording written while they had a ``joint state`` section of their own
    would otherwise open a second heading beside CUSTOM holding the same kind of thing. Retagged, not dropped —
    they are backend MEASUREMENTS (position/velocity/torque in display units), so a rebuild cannot recreate
    them and an old recording keeps the ones it has.
    """
    for t in payload["meta"]["tabs"]:
        if t["key"].startswith("state."):
            t["section"] = "custom"


def _add_custom_section(payload: dict) -> None:
    """Add the CUSTOM section: the live RL tab's overlay views (``target vs pos``) plus its val/SR counter.

    Synthesized at BUILD time, not sampled at record time, because both are VIEWS of series the log already
    wrote — so every existing recording gains them on a rebuild, and the payload carries no new measurement.
    The pairing plan comes from :func:`rl_monitor.overlay_plan`, the one place that knows which terms belong on
    one axis; values are that plan's own ``pick`` applied to the stored obs series, so the numbers are the same
    objects the obs tabs plot. val/SR carries no series at all — the page counts it off ``done``/``success``.
    """
    tabs = payload["meta"]["tabs"]
    # overlay_plan reads driver CARDS; an obs tab is the same (name, labels) pair, so hand it that view.
    cards = {"cards": [{"group": "obs", "name": t["key"].split(".", 1)[1], "labels": t["labels"]}
                       for t in tabs if t["key"].startswith("obs.")]}
    for ov in rl_monitor.overlay_plan(cards):
        # The chips are per JOINT (both series at once), so the marker key belongs in the unit line — it is the
        # only place left that says which of the pair is which. Suffixes come from the plan's own labels.
        pair = [lab.rsplit(" ", 1)[-1] for _i, lab in ov["rows"][0]["items"]]
        unit = f"{ov['unit']} · ○ {pair[0]} / ● {pair[-1]}"
        tabs.append({"key": ov["key"], "title": ov["title"], "unit": unit, "labels": ov["labels"],
                     "section": "custom", "line": ov["line"], "marker": ov["marker"], "rows": ov["rows"]})
        for env in payload["data"].values():
            env["series"][ov["key"]] = [env["series"][f"obs.{name}"][i] for name, i in ov["pick"]]
    tabs.append({"key": "eval.val/SR", "title": "Eval / SR", "unit": "cumulative to the playhead",
                 "labels": ["success", "attempts"], "section": "custom", "kind": "eval_sr"})


def build_report(record_dir: str, out_path: str | None = None, title: str = "") -> str:
    """Write the report for one recording directory (``data.json`` + its ``rollout.rrd``). Returns its path.

    ``record_dir`` is where the sim-service wrote the recording: the page addresses the ``.rrd`` by BASENAME,
    so the report must live in that same directory (and the whole directory is what gets uploaded).
    """
    t0 = time.perf_counter()
    data_path = os.path.join(record_dir, "data.json")
    assert os.path.exists(data_path), (
        f"no data.json in {record_dir} — the report needs the rollout log the sim-service writes beside the "
        f"recording (dashboard/rollout_log.py); was this recording made before that existed?")
    payload = json.loads(open(data_path).read())
    _state_channels_under_custom(payload)
    _add_custom_section(payload)
    meta = payload["meta"]
    # The rerun recording drives the 3D pane. Optional (newton + --rrd): without one the pane says so and the
    # plots still work, which is what a genesis recording gets until that spoke grows a rerun viewer.
    meta["rrd"] = next((os.path.basename(p)
                        for p in sorted(glob.glob(os.path.join(record_dir, "*.rrd")))), "")
    out = out_path or os.path.join(record_dir, "report.html")
    # </script> and <!-- inside a JSON string would end the script block early; < cannot.
    blob = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    page = _PAGE.replace("__PAYLOAD__", blob).replace(
        "__TITLE__", (title or f"{meta['task']} · {meta['engine']} · eval rollout").replace("<", "")
    ).replace("__RERUN_BASE__", _RERUN_BASE)
    with open(out, "w") as f:
        f.write(page)
    print(f"[rollout-report] {len(meta['envs'])} envs × {len(meta['tabs'])} channels "
          f"({os.path.getsize(out) / 1e6:.1f} MB in {time.perf_counter() - t0:.1f}s) -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record_dir", help="recording dir holding data.json + rollout.rrd")
    ap.add_argument("--out", default=None, help="output HTML (default <record_dir>/report.html)")
    ap.add_argument("--title", default="", help="page title (default '<task> · <engine> · eval rollout')")
    a = ap.parse_args()
    build_report(a.record_dir, a.out, a.title)


if __name__ == "__main__":
    main()
