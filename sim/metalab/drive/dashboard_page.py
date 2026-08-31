"""Standalone dashboard page — the single HTML/CSS/JS source served by BOTH servers.

Control + live-plot dashboard. HEADER: everything that acts on the RUN as a whole — status, the simulator
transport (Play/Pause/Stop, where Pause halts physics itself) and the Position/Torque toggle — plus the
SUB-TABS, which swap the whole body:

* **Trajectory** — the scrollable list of every discovered CSV group (click highlights a row, double-click
  selects it; the header's Play is what starts playback);
* **Monitor** — the plot viewer: a PLOT TAB per monitor channel (/describe channels) over one live line
  chart, with the active tab's per-series checkboxes in the left column. The ONLY view with the chart;
* **Joint Control** — one row per drivable joint: take-over checkbox, name, ROM-bounded slider and a
  degree field. Checked joints have their PD target written from here (``joint_target``), overriding the
  trajectory on those axes.

Snapshots keep filling every channel buffer whichever sub-tab is open (the stream is never torn down), so
returning to Monitor shows an unbroken timeline — only the *drawing* pauses while the chart is unmounted.

Styled to match the Launchpad console tab (same palette / .seg buttons; light default + dark via
prefers-color-scheme). Plotting mirrors sim/_runtime/telemetry.py (canvas, per-series toggles, legend,
hover crosshair) but over a rolling window — standalone has no fixed episode length.

Every snapshot is appended to EVERY channel's buffer on a single shared timeline, whichever tab is open —
so all tabs stay in lockstep and switching tabs never shows a hole. On `paused` the page stops appending
(plots freeze at the pause instant) so you can flip through the tabs and read the same frozen step off
each one; appending resumes on Resume.

Kept in a module with **zero imports** so the stdlib-only Launchpad (``launchpad/server.py``, which never
has an engine env) can load this file by path and serve the very same page as an offline PREVIEW — the
Standalone tab shows its window even with no run, so a page edit is visible without booting a sim.
``drive/control_server.py`` serves it live from inside the sim runner. The page therefore addresses its
endpoints RELATIVELY (``describe`` / ``stream`` / ``cmd``), working both at the runner's ``/`` and under
the Launchpad's ``/simui/`` prefix; ``describe.preview`` marks the no-run case (no stream, controls off).
"""

# Where the standalone runner caches its ``/describe`` payload (repo-relative) so the Launchpad's offline
# preview can render the LAST run's real channel set with no engine env. Written by runtime/standalone.py,
# read by launchpad/server.py — declared here because this is the one module both sides can load.
DESCRIBE_CACHE_REL = "logs/standalone/last_describe.json"

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MetaLab · trajectory</title>
<style>
:root{--ground:#eef1f4;--surface:#fff;--surface2:#e6ebef;--ink:#131a21;--soft:#4c5964;--faint:#7a8894;
--line:#d3dae0;--line2:#b7c1c9;--accent:#0f7d8c;--accent-ink:#0a5a66;--signal:#b06a2c;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif}
@media(prefers-color-scheme:dark){:root{--ground:#0b1015;--surface:#121a21;--surface2:#18222b;--ink:#dde5eb;
--soft:#96a4af;--faint:#67757f;--line:#233039;--line2:#35434e;--accent:#34aebd;--accent-ink:#6fcdd9;--signal:#cf925f}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px;display:flex;flex-direction:column}
.top{padding:12px 16px 0;border-bottom:1px solid var(--line)}
.tophd{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.rlseg{margin-left:auto}
h1{font-size:.95rem;font-weight:750;margin:0;flex:none}h1 span{color:var(--accent-ink)}
/* sub-tabs: which view the body shows (the plot viewer belongs to Monitor only) */
.stabs{display:flex;gap:2px;margin-top:9px}
.stab{font:inherit;font-size:.86rem;padding:8px 15px;border:none;background:none;color:var(--soft);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.stab:hover{color:var(--ink)}
.stab.on{color:var(--accent-ink);border-bottom-color:var(--accent);font-weight:700}
.layout{flex:1;min-height:0;display:flex;gap:14px;padding:12px 16px 14px}
.controls{flex:0 0 19rem;min-width:0;display:flex;flex-direction:column;min-height:0}
/* plot viewer hidden (Trajectory / Joint Control) → the controls take the freed width */
.layout.solo .controls{flex:1 1 auto}
.view{flex:1;min-height:0;display:flex;flex-direction:column;gap:12px}
/* Trajectory: transport on the left, the scrollable group list filling the rest */
#v-traj{flex-direction:row;gap:16px}
.tplay{flex:0 0 19rem;display:flex;flex-direction:column;gap:9px}
.glwrap{flex:1;min-width:0;display:flex;flex-direction:column;gap:6px}
.glist{flex:1;min-height:5rem;overflow:auto;display:flex;flex-direction:column;gap:2px;
border:1px solid var(--line);border-radius:8px;padding:6px;background:var(--surface)}
.glist.dis{opacity:.5;pointer-events:none}
.grow{display:flex;align-items:baseline;gap:9px;padding:6px 9px;border-radius:6px;cursor:pointer;
font-family:var(--mono);font-size:.74rem;border:1px solid transparent;user-select:none}   /* dblclick=play, not select-word */
.grow:hover{background:var(--surface2)}
.grow.mark{background:var(--surface2);border-color:var(--line2);border-style:dashed}   /* single click: highlight only */
.grow.on{background:var(--surface2);border-color:var(--accent);border-style:solid;color:var(--accent-ink);font-weight:700}
.grow .gdir{color:var(--faint);font-size:.66rem;font-weight:400}
.grow .gn{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gsel{font-family:var(--mono);font-size:.74rem;color:var(--accent-ink);word-break:break-all}
/* Joint Control: one row per drivable joint — [check] name [lo] slider [hi] [deg field] */
.jclist{flex:1;min-height:6rem;overflow:auto;display:flex;flex-direction:column;gap:1px;
border:1px solid var(--line);border-radius:8px;padding:6px;background:var(--surface)}
.jcrow{display:grid;grid-template-columns:auto minmax(9rem,15rem) 3.6rem minmax(6rem,1fr) 3.6rem 5.2rem;
align-items:center;gap:9px;padding:3px 7px;border-radius:5px;font-family:var(--mono);font-size:.72rem}
.jcrow:hover{background:var(--surface2)}
.jcrow.on .jcn{color:var(--accent-ink);font-weight:700}
.jcrow input[type=checkbox]{margin:0;cursor:pointer}
.jcn{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--soft)}
.jcend{font-size:.64rem;color:var(--faint);text-align:right}
.jcend.hi{text-align:left}
.jcend.unb{color:var(--signal)}                      /* model declares no limit → ±180 fallback span */
.jcrow input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer;margin:0}
.jcrow input[type=number]{font:inherit;font-family:var(--mono);font-size:.72rem;width:100%;padding:3px 5px;
text-align:right;border:1px solid var(--line2);border-radius:5px;background:var(--surface);color:var(--ink)}
.jcrow input:disabled{opacity:.45;cursor:default}
.view[hidden],.plots[hidden]{display:none}
.plots{flex:1;min-width:0;display:flex;flex-direction:column;gap:9px}
@media(max-width:780px){.layout{flex-direction:column}.controls{flex:none}.plots{min-height:64vh}}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.lbl{font-family:var(--mono);font-size:.6rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);width:100%}
button{font:inherit;font-size:.86rem;background:var(--surface);color:var(--ink);border:1px solid var(--line2);border-radius:7px;padding:7px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.45;cursor:default}
.seg{display:inline-flex;border:1px solid var(--line2);border-radius:7px;overflow:hidden}
.seg button{border:none;border-right:1px solid var(--line);border-radius:0;background:var(--surface);color:var(--soft);padding:7px 14px}
.seg button:last-child{border-right:none}
.seg button.on{background:var(--signal);color:#fff;font-weight:700}
.status{font-family:var(--mono);font-size:.72rem;color:var(--soft);word-break:break-word;flex:1;min-width:9rem}
.status .froz{color:var(--signal);font-weight:700}
/* run-health dot: green = live and running on the gains it loaded, amber = robot_model.json has been
   edited since (pending a reset), grey = nothing streaming (preview / dead run) */
.sdot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--faint);
margin-right:7px;vertical-align:middle;transition:background .2s}
.sdot.ok{background:#2eb872}
.sdot.warn{background:#e0a300;box-shadow:0 0 0 3px rgba(224,163,0,.20)}
.status .gwarn{color:var(--signal);font-weight:700}
#moderow{flex:none}#moderow .lbl{width:auto}
.transport{flex:none;gap:6px}   /* sim transport: header-level, left of the Mode segment */
.note{font-size:.74rem;line-height:1.55;color:var(--faint)}
/* Joint Control: not built yet — a dashed panel that says so instead of a silently empty column */
.ph{flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center;gap:9px;padding:14px;
background:var(--surface);border:1px dashed var(--line2);border-radius:8px;color:var(--soft);font-size:.76rem;line-height:1.55}
.ph b{font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--signal)}
.ptabs{flex:none;display:flex;gap:5px;flex-wrap:wrap;align-items:center}
table.srt{border-collapse:collapse;margin:10px 14px 14px;font-family:var(--mono);font-size:.78rem}
table.srt th,table.srt td{padding:7px 16px;border-bottom:1px solid var(--line);text-align:right}
table.srt thead th{font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);
font-weight:600}
table.srt tbody th{text-align:left;color:var(--accent-ink)}
table.srt tr.tot th,table.srt tr.tot td{border-bottom:none;border-top:2px solid var(--line2)}
.ptabs.sel2{gap:7px}
.csel{font-family:var(--mono);font-size:.74rem;padding:5px 9px;border:1px solid var(--line2);
border-radius:7px;background:var(--surface);color:var(--ink);cursor:pointer}
.csel:focus{outline:2px solid var(--accent);outline-offset:1px}
#psec{text-transform:uppercase;letter-spacing:.05em;color:var(--soft)}
.ptab{font-family:var(--mono);font-size:.72rem;padding:5px 12px;border:1px solid var(--line2);border-radius:7px;background:var(--surface);color:var(--soft);cursor:pointer}
.ptab:hover{border-color:var(--accent)}
.ptab.on{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:700}
.jhd{display:flex;align-items:center;gap:6px}.jhd .lbl{width:auto;flex:1}.jhd button{padding:3px 9px;font-size:.68rem}
.jointsel{flex:1;min-height:4rem;overflow:auto;display:flex;flex-direction:column;gap:1px;border:1px solid var(--line);border-radius:7px;padding:5px;background:var(--surface)}
.chip{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:.7rem;padding:2px 4px;border-radius:4px;cursor:pointer}
.chip:hover{background:var(--surface2)}.chip input{margin:0;cursor:pointer}
.sw{width:10px;height:10px;border-radius:2px;flex:none}
.chip .cn{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* grouped selector (Joint Kp): joint heading + that joint's K-row entries under it */
.jgrp{padding:3px 2px 4px;border-bottom:1px solid var(--line)}
.jgrp:last-child{border-bottom:none}
.jgn{font-family:var(--mono);font-size:.68rem;color:var(--accent-ink);font-weight:700;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.jgi{display:flex;flex-wrap:wrap;gap:2px 8px;padding-left:6px}
.jgi .chip{padding:1px 3px}
/* one plot per checked series (Joint Kp: per joint) — a shared axis across scales is unreadable, so each
   pane autoscales itself. Grid shape is set inline from the pane count: 1x1, 1x2, then 2xN up to 2x5. */
.panes{flex:1;min-height:0;display:grid;gap:9px;
grid-template-columns:repeat(var(--pc,1),minmax(0,1fr));grid-template-rows:repeat(var(--pr,1),minmax(0,1fr))}
.card{min-height:0;min-width:0;display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.pempty{flex:1;display:flex;align-items:center;justify-content:center;padding:20px;text-align:center;
color:var(--faint);font-size:.8rem;line-height:1.6}
.cardhd .ov{color:var(--signal);font-weight:700;font-size:.68rem}
/* channel title + unit once for the whole grid — repeating it in ten pane headers just crowds them out */
.panehd{flex:none;font-family:var(--mono);font-size:.72rem;font-weight:700;color:var(--accent-ink)}
.panehd span{color:var(--faint);font-weight:400}
.cardhd{font-family:var(--mono);font-size:.7rem;font-weight:700;color:var(--accent-ink);padding:5px 9px;border-bottom:1px solid var(--line);background:var(--surface2);
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cardhd span{color:var(--faint);font-weight:400}
.chartwrap{position:relative;flex:1;min-height:0}
canvas{display:block;width:100%;height:100%}
.legend{position:absolute;top:4px;right:6px;display:flex;flex-direction:column;gap:1px;font-family:var(--mono);font-size:.68rem;
border:1px solid var(--line);border-radius:5px;padding:4px 7px;max-width:52%;max-height:82%;overflow:hidden;pointer-events:none;background:var(--surface);opacity:.94}
.lg{display:flex;align-items:center;gap:5px;white-space:nowrap}.lg b{margin-left:auto}
.sw2{width:8px;height:8px;border-radius:2px;flex:none}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--accent);border-radius:5px;padding:5px 8px;font-family:var(--mono);font-size:.68rem;z-index:5;white-space:nowrap;max-height:70%;overflow:hidden}
.tip .th{color:var(--faint);margin-bottom:2px}
</style></head><body>
<div class="top">
  <div class="tophd">
    <h1>MetaLab · <span>Standalone</span></h1>
    <div class="status" id="status"><span class="sdot" id="sdot"></span><span id="stxt">—</span></div>
    <!-- RL only: freezes the SIMULATOR (physics), the same hold the viewer's own Pause applies. Both states
         are always visible so which one is active is readable without remembering what the label means. -->
    <span class="seg rlseg" id="rlpause" style="display:none"
          title="Freeze the simulator (physics stops; the rollout waits). The viewer's own Pause and Space do the same. Paused plots pan (drag) and zoom (wheel).">
      <button data-p="1">⏸ Pause</button><button data-p="0">▶ Play</button></span>
    <div class="bar transport"
         title="Simulator transport — Pause freezes physics itself (sim time stops), Play runs it (and starts the selected trajectory group when nothing is playing), Stop resets to the init pose and stays frozen">
      <button id="play">▶ Play</button><button id="pause">⏸ Pause</button><button id="stop">■ Stop</button></div>
    <div class="bar" id="moderow" style="display:none"><span class="lbl">Mode</span>
      <span class="seg" title="Position: gravcomp + PD tracks target (trajectory plays here) · Torque: PD off, floats on gravity feedforward — coupled joints via the motor-torque clamp (real motor limits), passive joints (waist) plain gravcomp">
        <button id="mpos">Position</button><button id="mtq">Torque</button></span></div>
  </div>
  <div class="stabs" id="stabs">
    <button class="stab on" data-v="traj">Trajectory</button>
    <button class="stab" data-v="mon">Monitor</button>
    <button class="stab" data-v="jc">Joint Control</button>
  </div>
  <!-- RL only: one tab per published env. Every env is buffered, so switching keeps its history. -->
  <div class="stabs" id="etabs" style="display:none"></div>
</div>
<div class="layout" id="layout">
  <div class="controls">
    <section class="view" id="v-traj">
      <div class="tplay">
        <div class="lbl">Selected group</div>
        <div class="gsel" id="gsel">— none —</div>
        <div class="note">Pick a group on the right — click highlights, double-click selects. Playback
          starts only from Play (top bar), which streams the selected group's CSVs as joint targets
          (cubic-Hermite, 1 s ramp-in from the current pose); joints the group does not name keep holding
          their init pose. Play on a paused run resumes it instead of restarting the group. No playback in
          Torque mode.</div>
      </div>
      <div class="glwrap">
        <div class="lbl" id="glhd">Trajectory groups</div>
        <div class="glist" id="glist"></div>
      </div>
    </section>
    <section class="view" id="v-mon" hidden>
      <div class="jhd"><span class="lbl" id="serhd">Series</span><button id="selall">all</button><button id="selnone">none</button></div>
      <div class="jointsel" id="serlist"></div>
    </section>
    <section class="view" id="v-jc" hidden>
      <div class="jhd"><span class="lbl" id="jchd">Joints</span>
        <button id="jcnone">release all</button></div>
      <div class="jclist" id="jclist"></div>
      <div class="note">Check a joint to take it over: its target is seeded from the joint's current
        position (no jump) and then follows the slider / the degree field. Unchecked joints are left to the
        trajectory or their init pose; a hand-driven joint overrides the trajectory on its own axis.
        Degrees here, radians in the engine. Targets are PD setpoints — a joint that is fighting gravity or
        a contact will sit short of its commanded angle, which is what the Monitor tab is for. In Torque
        mode the gravity-compensated joints float and ignore these targets.</div>
    </section>
  </div>
  <div class="plots" id="plots">
    <div class="ptabs" id="ptabs"></div>
    <div class="panehd" id="panehd"></div>
    <div class="panes" id="panes"></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const PALETTE=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#00b8b8","#f032e6","#9a6324",
  "#469990","#808000","#e6beff","#fabed4","#42d4f4","#ffd8b1","#000075","#a9a9a9"];
const col=i=>PALETTE[i%PALETTE.length];
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim()||"#888";
const fmt=(x,d)=>(typeof x==="number"&&isFinite(x))?x.toFixed(d===undefined?2:d):String(x);
const short=j=>String(j).replace(/_Joint$/,"");
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// ---- sub-tabs ---------------------------------------------------------------
// Trajectory / Monitor / Joint Control swap the body. The plot viewer (tabs + chart) belongs to Monitor
// alone; the other two get the freed width (.solo). Buffers keep filling while it is unmounted (onSnap is
// independent of the DOM) — only draw() idles, since a hidden canvas has no size. Re-entering Monitor
// redraws the full buffer, so the timeline has no hole.
const VIEWS=["traj","mon","jc"];
function showView(v){
  document.querySelectorAll(".stab").forEach(b=>b.classList.toggle("on",b.dataset.v===v));
  VIEWS.forEach(k=>{$("v-"+k).hidden=(k!==v);});
  $("plots").hidden=(v!=="mon");
  $("layout").classList.toggle("solo",v!=="mon");
  if(v==="mon")scheduleDraw();
}
document.querySelectorAll(".stab").forEach(b=>b.onclick=()=>showView(b.dataset.v));
showView("traj");                                 // open on the player; markup defaults stay in one place

// ---- control channel (up) ---------------------------------------------------
// Endpoints are RELATIVE to this page's directory: the runner serves it at "/" (→ /describe …), the
// Launchpad serves the identical page at "/simui/" as an offline preview (→ /simui/describe …).
async function post(b){try{await fetch("cmd",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});}catch(_){}}
// Simulator transport (header): Play runs the sim and — only when nothing is mid-playback — starts the
// selected group, so it doubles as "resume". Pause halts physics itself; Stop resets and stays frozen.
// Never disabled on a live run: with no group selected Play is still the way to unfreeze a paused sim.
$("play").onclick =()=>post({cmd:"play",group:GROUP});
$("pause").onclick=()=>post({cmd:"pause"});
$("stop").onclick =()=>post({cmd:"stop"});
function setMode(tq){TORQUE=tq;$("mtq").classList.toggle("on",tq);$("mpos").classList.toggle("on",!tq);syncPlay();}

// ---- trajectory group list ---------------------------------------------------
// The whole discovered set as a scrollable row list (was a combobox): one mouse-scroll shows every group
// instead of a dropdown that hides them. Single click only HIGHLIGHTS a row (dashed, nothing committed),
// double click SELECTS it (solid accent + the Selected group readout). Playback is Play's job alone — no
// click path starts the robot moving. Play stays disabled until a group is selected; the old combobox let
// an empty selection reach the runner as a failed play.
let GROUP=null,MARK=null,TORQUE=false,PREVIEW=false;
function syncPlay(){$("play").disabled=PREVIEW;}   // transport is sim-wide: only a dead run disables it
function buildGroups(gs){
  $("glhd").textContent=`Trajectory groups · ${gs.length}`;
  // label is "<parent dir>/<name>_group" — show the dir dim, the trimmed name as the row, full path on hover
  $("glist").innerHTML=gs.map(g=>{const p=String(g.label).split("/");
    return `<div class="grow" data-p="${esc(g.path)}" title="${esc(g.path)}">`
      +`<span class="gdir">${esc(p.slice(0,-1).join("/"))}/</span>`
      +`<span class="gn">${esc(p[p.length-1].replace(/_group$/,""))}</span></div>`;}).join("");
  $("glist").querySelectorAll(".grow").forEach((el,i)=>{
    el.onclick=()=>{MARK=gs[i].path;paintRows();};      // highlight only — nothing is committed
    el.ondblclick=()=>selectGroup(gs[i]);});
  syncPlay();
}
function paintRows(){$("glist").querySelectorAll(".grow").forEach(el=>{
  el.classList.toggle("on",el.dataset.p===GROUP);
  el.classList.toggle("mark",el.dataset.p===MARK&&el.dataset.p!==GROUP);});}
function selectGroup(g){
  GROUP=MARK=g.path; $("gsel").textContent=g.label;
  paintRows(); syncPlay();
}
$("mpos").onclick=()=>{setMode(false);post({cmd:"mode",torque:false});};
$("mtq").onclick =()=>{setMode(true); post({cmd:"mode",torque:true});};

// ---- joint control (manual per-joint targets) --------------------------------
// One row per drivable joint from /describe `controls` — the same set the runner writes PD targets for and
// the same set the Monitor tab plots, so a row and its trace line up. DEGREES throughout the page (slider,
// field, ROM ends); the runner converts to radians. Checking a row seeds it from the joint's LIVE position
// (LIVEPOS, off the joint_pos channel) so taking over never snaps the joint; unchecking releases it.
// Every change posts the FULL checked set — the runner replaces its hand-driven set wholesale, so a release
// needs no separate command. Slider drags are coalesced (one post per frame-ish) instead of per pixel.
let CTRL=[],JCVAL={},JCON={},LIVEPOS={},_jcTimer=0;
const jcRound=v=>Math.round(v*10)/10;
function buildJointControl(cs){
  CTRL=cs;
  $("jchd").textContent=`Joints · ${cs.length}`;
  if(!cs.length){   // pre-`controls` describe (an old cached preview) — say so rather than show an empty box
    $("jclist").innerHTML='<div class="note">This run reported no drivable joints '
      +'(a describe payload from before Joint Control existed).</div>';
    return;
  }
  $("jclist").innerHTML=cs.map((c,i)=>{
    JCVAL[c.name]=c.init; JCON[c.name]=false;
    const u=c.bounded?"":" unb";                     // unbounded axis: the ±180 span is ours, flag both ends
    return `<div class="jcrow" data-i="${i}">`
      +`<input type="checkbox" class="jcen">`
      +`<span class="jcn" title="${esc(c.name)}">${esc(short(c.name))}</span>`
      +`<span class="jcend${u}">${fmt(c.lo,1)}°</span>`
      +`<input type="range" class="jcsl" min="${c.lo}" max="${c.hi}" step="0.1" value="${c.init}" disabled>`
      +`<span class="jcend hi${u}">${fmt(c.hi,1)}°</span>`
      +`<input type="number" class="jcin" min="${c.lo}" max="${c.hi}" step="0.1" value="${fmt(c.init,1)}" disabled>`
      +`</div>`;}).join("");
  $("jclist").querySelectorAll(".jcrow").forEach(row=>{
    const c=CTRL[+row.dataset.i],ck=row.querySelector(".jcen"),
          sl=row.querySelector(".jcsl"),nu=row.querySelector(".jcin");
    ck.onchange=()=>{
      JCON[c.name]=ck.checked;
      // take over AT the joint's current angle (live value if the stream has one), so nothing jumps
      if(ck.checked)jcSet(c,sl,nu,LIVEPOS[c.name]!==undefined?LIVEPOS[c.name]:JCVAL[c.name]);
      row.classList.toggle("on",ck.checked); sl.disabled=nu.disabled=!ck.checked||PREVIEW;
      sendJoints();
    };
    sl.oninput=()=>{jcSet(c,sl,nu,+sl.value);sendJoints();};
    nu.onchange=()=>{jcSet(c,sl,nu,+nu.value);sendJoints();};   // typed value is clamped to the ROM by jcSet
  });
}
function jcSet(c,sl,nu,v){
  v=jcRound(Math.min(c.hi,Math.max(c.lo,isFinite(v)?v:c.init)));
  JCVAL[c.name]=v; sl.value=v; nu.value=fmt(v,1);
}
function sendJoints(){                               // coalesce a drag into one post per ~50 ms
  if(PREVIEW||_jcTimer)return;
  _jcTimer=setTimeout(()=>{_jcTimer=0;
    const targets={}; CTRL.forEach(c=>{if(JCON[c.name])targets[c.name]=JCVAL[c.name];});
    post({cmd:"joint_target",targets});},50);
}
$("jcnone").onclick=()=>{                            // release every joint back to trajectory / init pose
  $("jclist").querySelectorAll(".jcrow").forEach(row=>{
    const c=CTRL[+row.dataset.i],ck=row.querySelector(".jcen");
    ck.checked=false; JCON[c.name]=false; row.classList.remove("on");
    row.querySelector(".jcsl").disabled=row.querySelector(".jcin").disabled=true;});
  sendJoints();
};

// ---- channels (= plot tabs) + rolling buffers --------------------------------
// One tab per /describe channel. EVERY channel is buffered on the shared XS timeline on every snapshot,
// regardless of which tab is open — so tabs are always time-aligned and a tab switch never shows a hole.
let CH=[],ACT=null;                               // channel meta [{key,title,unit,labels,digits}] + active key
const SEL={},BUF={};        // SEL: key -> Set(checked dim) ; BUF: env -> key -> [per-dim value arrays]
// One buffer set PER ENV so switching envs keeps each one's history (standalone publishes a single env "0").
let ENV="0", ENVS=["0"];
let PAUSED=false;                                 // sim frozen → plots become pan/zoom-able (p.vw)
let PAUSE_ECHO=0;                                 // ignore incoming `paused` until this time (see setPauseUi)
let DRAG=null;                                    // {p, x, y, vw} while a pane is being dragged
let EVAL_SR=null;                                 // [[success, attempts], ...] by env — a TABLE, not a series
const bufOf=k=>(BUF[ENV]||{})[k]||[];
const CAP=600;                                    // rolling window: samples kept per series
let XS=[],tick=0,HOVER=null,_raf=0;
const chan=k=>CH.find(c=>c.key===k);
const scheduleDraw=()=>{if(_raf)return;_raf=requestAnimationFrame(()=>{_raf=0;draw();});};
function resetBuf(){XS=[];tick=0;ENVS.forEach(e=>{BUF[e]={};CH.forEach(c=>{BUF[e][c.key]=c.labels.map(()=>[]);});});}
function setEnvs(list){
  const next=(list&&list.length?list:["0"]).map(String);
  if(next.join()===ENVS.join())return;
  ENVS=next; if(!ENVS.includes(ENV))ENV=ENVS[0];
  $("etabs").style.display=ENVS.length>1?"":"none";
  $("etabs").innerHTML=ENVS.map(e=>`<button class="stab${e===ENV?" on":""}" data-e="${e}">env ${e}</button>`).join("");
  $("etabs").querySelectorAll(".stab").forEach(b=>b.onclick=()=>selectEnv(b.dataset.e));
  resetBuf();
}
function selectEnv(e){
  if(!BUF[e])return; ENV=e;
  $("etabs").querySelectorAll(".stab").forEach(b=>b.classList.toggle("on",b.dataset.e===e));
  // Point the 3D viewer at the same env, so the plots and what you are watching agree. The driver dispatches
  // this to backend.focus_env: both spokes move the camera onto that env's tile (genesis scene.envs_offset,
  // newton viewer.world_offsets — same env_spacing grid), every env stays on screen.
  post({focus:+e});
  scheduleDraw();
}

fetch("describe").then(r=>r.json()).then(d=>{
  buildGroups(d.groups||[]);
  buildJointControl(d.controls||[]);
  // RL sends neither trajectory groups nor joint-control rows (the policy drives), so those sub-tabs have
  // nothing behind them: hide them and open on Monitor instead of an empty Trajectory list.
  // A sub-tab with nothing behind it is hidden either way (a standalone run may genuinely find no CSV group).
  if(!(d.groups&&d.groups.length))$("stabs").querySelector('[data-v="traj"]').style.display="none";
  if(!(d.controls&&d.controls.length))$("stabs").querySelector('[data-v="jc"]').style.display="none";
  setEnvs(d.num_envs?Array.from({length:d.num_envs},(_,i)=>String(i)):["0"]);
  // RL vs standalone keys on `num_envs`, which only the RL describe carries — NOT on "no trajectory groups",
  // which a standalone run with no CSVs would also satisfy, costing it its own transport.
  if(d.num_envs!==undefined&&d.num_envs!==null){
    showView("mon");
    // Monitor is RL's only view, so the sub-tab row is a one-item menu and the brand line says nothing the
    // env tabs do not. The transport and Position/Torque act on a standalone runner RL has none of.
    $("stabs").style.display="none";
    document.querySelectorAll(".tophd h1,.tophd .transport,#moderow").forEach(el=>{el.style.display="none";});
    $("rlpause").style.display="";       // the standalone transport is gone; this is RL's own sim hold
    $("rlpause").querySelectorAll("button").forEach(b=>b.onclick=()=>{
      const want=b.dataset.p==="1";
      // A snapshot published just BEFORE the sim read this command still says paused=false. Hold off on
      // incoming state briefly so that one in-flight frame cannot flip the button back under the click.
      PAUSE_ECHO=Date.now()+500; setPauseUi(want); post({pause:want});});
  }
  if(d.gravcomp!==null&&d.gravcomp!==undefined){$("moderow").style.display="";setMode(false);}  /* gravcomp active → Position/Torque row */
  CH=d.channels||[]; CH.forEach(c=>{SEL[c.key]=defaultSel(c);});
  resetBuf(); buildTabs(); showChannel(CH.length?CH[0].key:null);
  // preview = the Launchpad serving this page with NO standalone run behind it: tabs/checkboxes come from
  // the last run's channel set (if any) but there is nothing to stream and no runner to command.
  if(d.preview){setPreview(d);return;}
  connect();
}).catch(()=>{$("stxt").textContent="describe unavailable";});

function setPreview(d){
  PREVIEW=true;                                    // no runner to command: transport off, group list inert
  ["play","pause","stop","mpos","mtq","jcnone"].forEach(i=>{const el=$(i);if(el)el.disabled=true;});
  $("glist").classList.add("dis");
  $("jclist").querySelectorAll("input").forEach(el=>el.disabled=true);   // rows render, nothing is drivable
  $("stxt").innerHTML='<span class="froz">preview</span> — no standalone run. '
    +(CH.length?"tabs/series are from the last run; ":"")+"Launch Standalone mode for live data.";
}

// ---- plot tabs --------------------------------------------------------------
// Channels that declare a `section` (RL: actor obs / critic obs / reward / action) get TWO cascading
// selects — section, then channel within it. A flat row of 30+ tabs is unreadable, and the section is the
// thing you pick by first. Standalone declares none, so it keeps the flat tab row.
const sectionsOf=()=>[...new Set(CH.filter(c=>c.section).map(c=>c.section))];
const inSection=sec=>CH.filter(c=>c.section===sec);
function buildTabs(){
  const secs=sectionsOf();
  $("ptabs").classList.toggle("sel2",!!secs.length);
  if(!secs.length){        // standalone: one flat tab per channel
    $("ptabs").innerHTML=CH.map(c=>
      `<button class="ptab" data-k="${c.key}" title="${esc(c.unit||"")}">${esc(c.title)}</button>`).join("");
    $("ptabs").querySelectorAll(".ptab").forEach(el=>el.onclick=()=>showChannel(el.dataset.k));
    return;
  }
  $("ptabs").innerHTML=`<select id="psec" class="csel"></select><select id="pch" class="csel"></select>`;
  $("psec").innerHTML=secs.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join("");
  $("psec").onchange=()=>fillChannelSelect($("psec").value,true);
  $("pch").onchange=()=>showChannel($("pch").value);
  const cur=chan(ACT);
  $("psec").value=cur?cur.section:secs[0];
  fillChannelSelect($("psec").value,!cur);
}
function fillChannelSelect(sec,pick){
  const list=inSection(sec);
  $("pch").innerHTML=list.map(c=>
    `<option value="${esc(c.key)}"${c.unit?` title="${esc(c.unit)}"`:""}>`
    +`${esc(c.title)}${c.unit?` [${esc(c.unit)}]`:""}</option>`).join("");
  if(pick&&list.length)showChannel(list[0].key);
  else if(chan(ACT))$("pch").value=ACT;
}
function showChannel(k){
  ACT=k; HOVER=null;
  $("ptabs").querySelectorAll(".ptab").forEach(el=>el.classList.toggle("on",el.dataset.k===k));
  if($("pch")&&$("pch").value!==k)$("pch").value=k;
  const c=chan(k);
  if(!c){$("panes").innerHTML="";$("panehd").textContent="";$("serlist").innerHTML="";return;}
  $("panehd").innerHTML=`${esc(c.title)} <span>[${esc(c.unit)}]</span>`;
  if(isTable(c)){                                  // a counter table: no series to pick, every env at once
    $("serhd").textContent="Series · —";
    $("serlist").innerHTML='<div class="pempty">this view is a table over ALL envs — the env tabs and the '
      +'series checkboxes do not apply</div>';
    PANES=[]; HOVER=null; renderTable(); return;
  }
  $("serhd").textContent=`Series · ${c.labels.length}`;
  buildSeriesSel(c); buildPanes();
}

// ---- series selector (per tab; each tab remembers its own checkboxes) --------
// Every tab opens EMPTY — 48 joints (or 84 stiffness entries) drawn at once is noise, and which few
// series matter is the operator's question, not something a heuristic should guess. `all` fills a tab in
// one click; each tab remembers its own picks for the rest of the session.
function defaultSel(_c){return new Set();}
// Two selector shapes. Default: one chip per series (a joint set, a pose). Grouped (channel declares
// `rows`): a joint heading with that joint's entries beside it — Joint Kp plots a ROW of K per joint, so a
// flat list of 124 "A×B" names would hide which joint each belongs to.
const chip=(l,i,sel,txt)=>`<label class="chip"><input type="checkbox" data-i="${i}" ${sel.has(i)?"checked":""}>`
  +`<span class="sw" style="background:${col(i)}"></span>`
  +`<span class="cn" title="${esc(l)}">${esc(txt)}</span></label>`;
function buildSeriesSel(c){
  const sel=SEL[c.key];
  $("serlist").innerHTML=c.rows
    ? c.rows.map(r=>`<div class="jgrp"><div class="jgn" title="${esc(r.joint)}">${esc(r.joint)}</div>`
        +`<div class="jgi">${r.items.map(([i,tag])=>chip(c.labels[i],i,sel,tag)).join("")}</div></div>`).join("")
    : c.labels.map((l,i)=>chip(l,i,sel,short(l))).join("");
  $("serlist").querySelectorAll("input").forEach(el=>el.onchange=()=>{
    const i=+el.dataset.i; el.checked?sel.add(i):sel.delete(i); buildPanes();});
}
function syncChecks(){$("serlist").querySelectorAll("input").forEach(el=>el.checked=SEL[ACT].has(+el.dataset.i));}
$("selall").onclick =()=>{const c=chan(ACT);if(!c)return;c.labels.forEach((_,i)=>SEL[ACT].add(i));syncChecks();buildPanes();};
$("selnone").onclick=()=>{if(!chan(ACT))return;SEL[ACT].clear();syncChecks();buildPanes();};

// ---- live stream (down) -----------------------------------------------------
function connect(){
  const es=new EventSource("stream");
  es.onmessage=e=>{try{onSnap(JSON.parse(e.data));}catch(_){}};
  es.onerror=()=>{$("stxt").textContent="disconnected";$("sdot").className="sdot";};
}
function onSnap(s){
  // `paused` is the SIM's state (physics halted), so it outranks the playback state in the readout.
  const st=s.paused?"paused":(s.finished?"finished":s.playing?"playing":"running");
  // Motor gains are read from robot_model.json at build and swapped in on reset, never mid-step — so an
  // edit is reported here (amber dot) rather than applied silently under a moving robot.
  // Two different amber states: `gains_dirty` = the FILE moved ahead of the run (reset to apply);
  // `gain_warn` = the gains the run is ALREADY using are inconsistent (a differential group whose motors
  // no longer share a gain puts an off-diagonal term into K_q). The second one a reset will not fix.
  const dirty=!!s.gains_dirty, warn=(s.gain_warn||[]);
  $("sdot").className="sdot "+(dirty||warn.length?"warn":"ok");
  $("stxt").innerHTML=(warn.length?`<span class="gwarn" title="${esc(warn.join(" | "))}">`
      +`gain warning: ${esc(warn[0].split(":")[0])} motors do not share a gain — K_q is no longer `
      +`diagonal (hover for detail)</span> · `:"")
    +(dirty?'<span class="gwarn">motor gains edited — Reset (or Stop) to apply; '
      +'this run is still on the gains it started with</span> · ':"")
    +st+(s.playing?` · ${fmt(s.t,2)}/${fmt(s.duration,2)}s`:"")+(s.group?` · ${s.group}`:"")
    +(s.paused?' · <span class="froz">sim frozen</span> · drag to pan · wheel to zoom · double-click resets':"");
  // RL publishes {envs:{id:{key:[…]}}}, standalone a bare {ch:{key:[…]}} for its single env — one path.
  if(s.envs)setEnvs(Object.keys(s.envs));
  if(s.paused!==undefined&&$("rlpause").style.display!=="none"&&Date.now()>=PAUSE_ECHO)setPauseUi(!!s.paused);
  if(s.eval_sr){EVAL_SR=s.eval_sr; if(isTable(chan(ACT)))renderTable();}
  const ch=(s.envs?(s.envs[ENV]||{}):(s.ch||{}));
  // Live joint angles feed Joint Control's "take over where the joint is" seeding — kept fresh even while
  // paused (a frozen pose is still the pose a newly checked row must adopt).
  const jc=chan("joint_pos");
  if(jc&&ch.joint_pos)jc.labels.forEach((l,i)=>{LIVEPOS[l]=ch.joint_pos[i];});
  // Paused → stop appending: nothing is advancing, so every tab holds the pause instant and you can flip
  // through them reading the same frozen step. Play continues the timeline where it left off.
  if(!CH.length||s.paused)return;
  XS.push(tick++);
  // Append EVERY env, not just the drawn one: one snapshot carries them all, so the timeline (XS) is shared
  // and an env keeps its history while another is on screen.
  ENVS.forEach(e=>{const src=(s.envs?(s.envs[e]||{}):(s.ch||{})),tgt=BUF[e]||{};
    CH.forEach(c=>{const v=src[c.key]||[],b=tgt[c.key];if(b)for(let i=0;i<b.length;i++)b[i].push(v[i]);});});
  if(XS.length>CAP){XS.shift();ENVS.forEach(e=>CH.forEach(c=>{const b=(BUF[e]||{})[c.key];if(b)b.forEach(a=>a.shift());}));}
  scheduleDraw();
}

// ---- plot panes (one chart per checked series; Joint Kp: one per joint) ------
// A single overlaid axis is unreadable once the series differ in scale (4930 vs 0.8 N·m/rad), so each
// pane owns its own y autoscale. Grid grows 1x1 -> 1x2 -> 2x2 -> ... -> 2x5; MAXP is the ceiling, and
// anything past it is reported in the first pane's header rather than silently dropped.
const MAXP=10;
let PANES=[];                                     // [{title, idx:[series], cv, lg, tip}] for the open tab

// What shares a pane: a channel that declares `rows` groups by joint (Joint Kp — a joint's diag and its
// cross terms belong on ONE axis, they are the same row of K). Everything else = one series per pane,
// which is what makes a 7-dim pose readable (x/y/z in m next to a unit quaternion).
function paneGroups(c){
  const sel=SEL[c.key];
  if(c.rows) return c.rows.map(r=>({title:r.joint, idx:r.items.map(([i])=>i).filter(i=>sel.has(i))}))
                          .filter(p=>p.idx.length);
  return [...sel].sort((a,b)=>a-b).map(i=>({title:short(c.labels[i]), idx:[i]}));
}

// Optimistic on click, but every snapshot's `paused` is the authority — the viewer's own Pause / Space holds
// the sim as well, so the button must follow the SIM rather than only its own clicks.
function setPauseUi(on){
  PAUSED=on;
  $("rlpause").querySelectorAll("button").forEach(b=>b.classList.toggle("on",(b.dataset.p==="1")===on));
  // Leaving pause drops every hand-made view: the window is scrolling again, so a frozen x range would just
  // slide off screen. Re-entering pause starts from the live autoscale.
  if(!on){PANES.forEach(p=>{p.vw=null;});scheduleDraw();}
}

// Pane geometry and the autoscale window, factored out so the pan/zoom handlers and drawPane agree on both.
const geomOf=cv=>{const w=cv.clientWidth,h=cv.clientHeight,L=54,R=12,T=10,B=22;
  return {L:L,T:T,pw:w-L-R,ph:h-T-B};};
function curView(p){                              // the range drawPane would pick right now
  const c=chan(ACT), data=bufOf(c.key);
  const n=XS.length, x0=n?XS[0]:0, x1=n?XS[n-1]:1;
  let y0=Infinity,y1=-Infinity;
  p.idx.forEach(i=>{const a=data[i]||[];for(const v of a)if(isFinite(v)){if(v<y0)y0=v;if(v>y1)y1=v;}});
  if(!isFinite(y0)){y0=-1;y1=1;} if(y0===y1){y0-=1;y1+=1;}
  const pd=(y1-y0)*0.08;
  return {x0:x0, x1:x1>x0?x1:x0+1, y0:y0-pd, y1:y1+pd};
}
addEventListener("mouseup",()=>{if(DRAG){DRAG.p.cv.style.cursor="crosshair";DRAG=null;}});
addEventListener("blur",()=>{if(DRAG){DRAG.p.cv.style.cursor="crosshair";DRAG=null;}});

const isTable=c=>!!(c&&c.kind);
// val/SR: cumulative success/attempts per env for THIS run, plus the population row. Not a time series —
// counters only grow, so a plot would be a staircase and the number is what you want to read.
function renderTable(){
  const host=$("panes");
  host.style.removeProperty("--pc"); host.style.removeProperty("--pr");
  const rows=EVAL_SR||[];
  if(!rows.length){host.innerHTML='<div class="card"><div class="pempty">no episode has finished yet</div></div>';return;}
  const pct=(s,a)=>a?`${(100*s/a).toFixed(1)}%`:"—";
  let ts=0, ta=0;
  const body=rows.map((r,i)=>{const s=+r[0]||0,a=+r[1]||0;ts+=s;ta+=a;
    return `<tr><th>env ${i}</th><td><b>${s}</b> / ${a}</td><td>${pct(s,a)}</td></tr>`;}).join("");
  host.innerHTML='<div class="card"><div class="cardhd">Eval / SR · cumulative over this run</div>'
    +`<table class="srt"><thead><tr><th>ENV</th><th>SUCCESS / ATTEMPTS</th><th>SR</th></tr></thead>`
    +`<tbody>${body}<tr class="tot"><th>all</th><td><b>${ts}</b> / ${ta}</td><td>${pct(ts,ta)}</td></tr>`
    +`</tbody></table></div>`;
}

function buildPanes(){
  const c=chan(ACT), host=$("panes");
  PANES=[]; HOVER=null;
  if(!c){host.innerHTML="";return;}
  const all=paneGroups(c), list=all.slice(0,MAXP), hidden=all.length-list.length;
  if(!list.length){
    host.style.removeProperty("--pc"); host.style.removeProperty("--pr");
    host.innerHTML=`<div class="card"><div class="pempty">check a series on the left — each one gets its `
      +`own plot (up to ${MAXP}), so scales never share an axis</div></div>`;
    return;
  }
  const rows=list.length<=2?1:2, cols=Math.ceil(list.length/rows);
  host.style.setProperty("--pc",cols); host.style.setProperty("--pr",rows);
  host.innerHTML=list.map((p,k)=>`<div class="card"><div class="cardhd">${esc(p.title)}`
    +`${k===0&&hidden?` <b class="ov">+${hidden} not shown</b>`:""}</div>`
    +`<div class="chartwrap"><canvas></canvas><div class="legend"></div><div class="tip" hidden></div></div>`
    +`</div>`).join("");
  host.querySelectorAll(".card").forEach((el,k)=>{
    const p=list[k];
    p.cv=el.querySelector("canvas"); p.lg=el.querySelector(".legend"); p.tip=el.querySelector(".tip");
    p.vw=null;                                     // null = autoscale; set by pan/zoom while paused
    p.cv.addEventListener("mousemove",e=>{           // hover is per pane: crosshair only where the mouse is
      const r=p.cv.getBoundingClientRect();
      if(DRAG&&DRAG.p===p){                          // dragging pans instead of moving the crosshair
        const gm=geomOf(p.cv), vw=DRAG.vw;
        const dx=(e.clientX-DRAG.x)*(vw.x1-vw.x0)/Math.max(1,gm.pw);
        const dy=(e.clientY-DRAG.y)*(vw.y1-vw.y0)/Math.max(1,gm.ph);
        p.vw={x0:vw.x0-dx, x1:vw.x1-dx, y0:vw.y0+dy, y1:vw.y1+dy};   // +dy: screen y grows downward
        HOVER=null; scheduleDraw(); return;
      }
      HOVER={p:k, x:e.clientX-r.left}; scheduleDraw();});
    p.cv.addEventListener("mouseleave",()=>{HOVER=null;scheduleDraw();});
    p.cv.addEventListener("mousedown",e=>{
      if(!PAUSED)return;                             // a live window scrolls; panning it would fight the feed
      p.vw=p.vw||curView(p); DRAG={p:p, x:e.clientX, y:e.clientY, vw:{...p.vw}};
      p.cv.style.cursor="grabbing"; e.preventDefault();});
    p.cv.addEventListener("wheel",e=>{
      if(!PAUSED)return;
      const gm=geomOf(p.cv), r=p.cv.getBoundingClientRect(), vw=p.vw||curView(p);
      // Zoom about the CURSOR so the value under it stays put — otherwise zooming walks the point away.
      const fx=Math.min(1,Math.max(0,(e.clientX-r.left-gm.L)/Math.max(1,gm.pw)));
      const fy=Math.min(1,Math.max(0,(e.clientY-r.top-gm.T)/Math.max(1,gm.ph)));
      const k=Math.exp(e.deltaY*0.0015), ax=vw.x0+(vw.x1-vw.x0)*fx, ay=vw.y1-(vw.y1-vw.y0)*fy;
      p.vw={x0:ax-(ax-vw.x0)*k, x1:ax+(vw.x1-ax)*k, y0:ay-(ay-vw.y0)*k, y1:ay+(vw.y1-ay)*k};
      scheduleDraw(); e.preventDefault();},{passive:false});
    p.cv.addEventListener("dblclick",()=>{p.vw=null;scheduleDraw();});   // back to autoscale
    p.cv.style.cursor="crosshair";
    PANES.push(p);
  });
  scheduleDraw();
}

function draw(){
  const c=chan(ACT); if(!c||isTable(c))return;
  PANES.forEach((p,k)=>drawPane(c,p,(HOVER&&HOVER.p===k)?HOVER.x:null));
}

function drawPane(c,p,hx){
  const cv=p.cv,data=bufOf(c.key),dg=c.digits===undefined?2:c.digits,vis=p.idx;
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  if(!w||!h)return;
  cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext("2d");g.scale(dpr,dpr);g.clearRect(0,0,w,h);
  g.font="10px "+cssv("--mono");
  const L=54,R=12,T=10,B=22,pw=w-L-R,ph=h-T-B;
  let xMin,xMax,ymin,ymax;
  if(p.vw){                                        // hand-set by pan/zoom (paused only)
    xMin=p.vw.x0; xMax=p.vw.x1; ymin=p.vw.y0; ymax=p.vw.y1;
  }else{
    const n=XS.length;xMin=n?XS[0]:0;xMax=n?XS[n-1]:1;
    ymin=Infinity;ymax=-Infinity;
    vis.forEach(i=>{const a=data[i]||[];for(let k=0;k<a.length;k++){const v=a[k];if(isFinite(v)){if(v<ymin)ymin=v;if(v>ymax)ymax=v;}}});
    if(!isFinite(ymin)){ymin=-1;ymax=1;} if(ymin===ymax){ymin-=1;ymax+=1;}
    const pd=(ymax-ymin)*0.08;ymin-=pd;ymax+=pd;
  }
  const xr=Math.max(1e-9,xMax-xMin);
  const X=x=>L+pw*((x-xMin)/xr),Y=v=>T+ph*(1-(v-ymin)/(ymax-ymin));
  g.strokeStyle=cssv("--line");g.fillStyle=cssv("--faint");g.lineWidth=1;
  for(let k=0;k<=4;k++){const yy=T+ph*k/4;g.globalAlpha=.5;g.beginPath();g.moveTo(L,yy);g.lineTo(L+pw,yy);g.stroke();g.globalAlpha=1;
    g.textAlign="right";g.textBaseline="middle";g.fillText(fmt(ymax-(ymax-ymin)*k/4,Math.max(1,dg-1)),L-6,yy);}
  g.textAlign="center";g.textBaseline="top";
  // Vertical grid, same weight as the horizontal one — reading a step off a plot needs both axes ruled.
  for(let k=0;k<=5;k++){const xx=L+pw*k/5;
    g.strokeStyle=cssv("--line");g.globalAlpha=.5;g.beginPath();g.moveTo(xx,T);g.lineTo(xx,T+ph);g.stroke();g.globalAlpha=1;
    g.fillStyle=cssv("--faint");g.fillText(fmt(xMin+xr*k/5,0),xx,T+ph+5);}
  if(ymin<0&&ymax>0){const zy=Y(0);g.strokeStyle=cssv("--line2");g.globalAlpha=.9;g.beginPath();g.moveTo(L,zy);g.lineTo(L+pw,zy);g.stroke();g.globalAlpha=1;}
  // `line:false` = a SCATTER channel (the delay overlay): a connecting line interpolates between samples,
  // which is exactly the thing you must not read there — a transport delay is N discrete flat samples.
  const wantLine=c.line!==false, wantDots=c.marker||c.line===false;
  vis.forEach(i=>{const a=data[i]||[];if(!a.length)return;
    if(wantLine){g.strokeStyle=col(i);g.lineWidth=1.4;g.beginPath();
      let pen=false;                               // lift the pen over non-finite samples (gap, not a spike)
      for(let k=0;k<a.length;k++){const v=a[k];if(!isFinite(v)){pen=false;continue;}
        const px=X(XS[k]),py=Y(v);pen?g.lineTo(px,py):g.moveTo(px,py);pen=true;}
      g.stroke();}
    // One dot per SAMPLE: a step becomes countable, which is the point of the delay overlay. Skipped once the
    // dots would touch (they read as a thick line and cost draw time) — unless dots are ALL this channel has.
    if(wantDots&&(c.line===false||pw/Math.max(1,a.length)>=3.5)){g.fillStyle=col(i);
      const rad=c.line===false?2.2:1.9;
      for(let k=0;k<a.length;k++){const v=a[k];if(!isFinite(v))continue;
        g.beginPath();g.arc(X(XS[k]),Y(v),rad,0,6.2832);g.fill();}}});
  p.lg.innerHTML=vis.map(i=>{const a=data[i]||[],last=a.length?a[a.length-1]:NaN;
    return `<div class="lg"><span class="sw2" style="background:${col(i)}"></span><span>${short(c.labels[i])}</span><b>${fmt(last,dg)}</b></div>`;}).join("");
  drawHover(g,c,p,data,vis,dg,X,Y,L,T,pw,ph,cv,hx);
}

function drawHover(g,c,p,data,vis,dg,X,Y,L,T,pw,ph,cv,hx){
  const tipEl=p.tip;
  if(hx===null||!vis.length||!XS.length||hx<L||hx>L+pw){tipEl.hidden=true;return;}
  let ki=0,bd=Infinity;for(let k=0;k<XS.length;k++){const dd=Math.abs(X(XS[k])-hx);if(dd<bd){bd=dd;ki=k;}}
  const cx=X(XS[ki]);
  g.strokeStyle=cssv("--accent");g.globalAlpha=.6;g.lineWidth=1;g.beginPath();g.moveTo(cx,T);g.lineTo(cx,T+ph);g.stroke();g.globalAlpha=1;
  let rows=`<div class="th">#${XS[ki]}</div>`;
  vis.forEach(i=>{const v=(data[i]||[])[ki];
    if(isFinite(v)){g.fillStyle=col(i);g.beginPath();g.arc(cx,Y(v),3,0,7);g.fill();}
    rows+=`<div class="lg"><span class="sw2" style="background:${col(i)}"></span><span>${short(c.labels[i])}</span><b>${fmt(v,dg)}</b></div>`;});
  tipEl.innerHTML=rows;tipEl.hidden=false;
  const bw=cv.clientWidth;let tx=cx+12;if(tx>bw-tipEl.offsetWidth-6)tx=cx-tipEl.offsetWidth-12;
  tipEl.style.left=Math.max(4,tx)+"px";tipEl.style.top=(T+4)+"px";
}
window.addEventListener("resize",scheduleDraw);
</script></body></html>"""
