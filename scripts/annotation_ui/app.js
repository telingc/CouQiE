"use strict";
const $ = id => document.getElementById(id);
const colors = {0:[166,133,245],1:[255,65,77],2:[255,167,38],3:[0,190,220],4:[61,190,90],255:[155,163,175]};
const names = {0:"Background",1:"Penguin",2:"Other bird",3:"Seal",4:"Whale",255:"Ignore"};
const offline = Boolean(window.REFERENCE_BUNDLE);
const localPhotos = new Map();
let outputFolder = null;
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const labelCanvas = document.createElement("canvas"), labelCtx = labelCanvas.getContext("2d");
const state = {catalog:null,entry:null,image:null,ids:null,rgba:null,width:0,height:0,scale:1,tx:0,ty:0,tool:"brush",label:1,brush:30,opacity:.42,hide:false,space:false,dirty:false,polygon:[],drag:null,cursor:null,undo:[],redo:[],historyBytes:0,stroke:null,loading:false};
let drawPending=false;
function status(text,error=false){$("status").textContent=text;$("status").style.color=error?"#ff9d9d":"#b9c9e0";}
function dirty(){state.dirty=true;$("reviewed").checked=false;status("Unsaved changes");}
function requestDraw(){if(!drawPending){drawPending=true;requestAnimationFrame(()=>{drawPending=false;draw();});}}
function point(event){const r=canvas.getBoundingClientRect();return {x:(event.clientX-r.left-state.tx)/state.scale,y:(event.clientY-r.top-state.ty)/state.scale};}
function draw(){
  ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle="#0b1019";ctx.fillRect(0,0,canvas.width,canvas.height);
  if(!state.image)return;
  ctx.save();ctx.translate(state.tx,state.ty);ctx.scale(state.scale,state.scale);ctx.drawImage(state.image,0,0);
  if(!state.hide){ctx.globalAlpha=state.opacity;ctx.imageSmoothingEnabled=false;ctx.drawImage(labelCanvas,0,0);ctx.globalAlpha=1;}
  const line=1.5/state.scale;ctx.lineWidth=line;
  if($("guides").checked&&state.entry.id==="penguins_far"){
    ctx.strokeStyle="#ffffff";ctx.setLineDash([8/state.scale,5/state.scale]);
    for(const f of [.5,.625]){ctx.beginPath();ctx.moveTo(0,state.height*f);ctx.lineTo(state.width,state.height*f);ctx.stroke();}
    ctx.setLineDash([]);
  }
  const vertices=state.polygon;
  if(vertices.length){ctx.strokeStyle="#ffe576";ctx.fillStyle="#ffe576";ctx.beginPath();vertices.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));if(state.cursor)ctx.lineTo(state.cursor.x,state.cursor.y);ctx.stroke();for(const p of vertices){ctx.beginPath();ctx.arc(p.x,p.y,3/state.scale,0,Math.PI*2);ctx.fill();}}
  if(state.drag?.type==="rectangle"&&state.cursor){const a=state.drag.start,b=state.cursor;ctx.strokeStyle="#ffe576";ctx.strokeRect(a.x,a.y,b.x-a.x,b.y-a.y);}
  if(state.cursor&&state.tool==="brush"&&!state.space){ctx.strokeStyle="white";ctx.beginPath();ctx.arc(state.cursor.x,state.cursor.y,state.brush/2,0,Math.PI*2);ctx.stroke();}
  ctx.restore();$("coordinates").textContent=`${Math.round(state.scale*100)}%${state.cursor?` · x ${Math.floor(state.cursor.x)}, y ${Math.floor(state.cursor.y)}`:""}`;
}
function fit(){const margin=30;state.scale=Math.min((canvas.width-margin)/state.width,(canvas.height-margin)/state.height);state.tx=(canvas.width-state.width*state.scale)/2;state.ty=(canvas.height-state.height*state.scale)/2;requestDraw();}
new ResizeObserver(()=>{canvas.width=canvas.clientWidth;canvas.height=canvas.clientHeight;if(state.image)fit();}).observe($("workspace"));
function selectLabel(label){state.label=Number(label);document.querySelectorAll("[data-label]").forEach(b=>b.classList.toggle("active",Number(b.dataset.label)===state.label));}
function selectTool(tool){state.tool=tool;state.polygon=[];document.querySelectorAll("[data-tool]").forEach(b=>b.classList.toggle("active",b.dataset.tool===tool));canvas.style.cursor=tool==="pan"?"grab":"crosshair";requestDraw();}
for(const key of [0,1,2,3,4,255]){const b=document.createElement("button");b.className="class-button";b.dataset.label=key;b.innerHTML=`<span class="swatch" style="background:rgb(${colors[key]})"></span>${names[key]} <kbd>${key===255?"I":key}</kbd>`;b.onclick=()=>selectLabel(key);$("classes").appendChild(b);}
document.querySelectorAll("[data-tool]").forEach(b=>b.onclick=()=>selectTool(b.dataset.tool));selectLabel(1);
function updatePixels(x0,y0,x1,y1){
  x0=Math.max(0,Math.floor(x0));y0=Math.max(0,Math.floor(y0));x1=Math.min(state.width,Math.ceil(x1));y1=Math.min(state.height,Math.ceil(y1));
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const i=y*state.width+x,c=colors[state.ids[i]],q=i*4;state.rgba.data[q]=c[0];state.rgba.data[q+1]=c[1];state.rgba.data[q+2]=c[2];state.rgba.data[q+3]=255;}
  if(x1>x0&&y1>y0)labelCtx.putImageData(state.rgba,0,0,x0,y0,x1-x0,y1-y0);requestDraw();
}
const tileSize=128;
function copyTile(tile){const out=new Uint8Array(tile.w*tile.h);for(let y=0;y<tile.h;y++)out.set(state.ids.subarray((tile.y+y)*state.width+tile.x,(tile.y+y)*state.width+tile.x+tile.w),y*tile.w);return out;}
function beginStroke(){state.stroke=new Map();}
function capture(x0,y0,x1,y1){
  for(let ty=Math.floor(Math.max(0,y0)/tileSize);ty<=Math.floor(Math.min(state.height-1,y1)/tileSize);ty++)for(let tx=Math.floor(Math.max(0,x0)/tileSize);tx<=Math.floor(Math.min(state.width-1,x1)/tileSize);tx++){
    const key=`${tx},${ty}`;if(!state.stroke.has(key)){const tile={x:tx*tileSize,y:ty*tileSize,w:Math.min(tileSize,state.width-tx*tileSize),h:Math.min(tileSize,state.height-ty*tileSize)};tile.before=copyTile(tile);state.stroke.set(key,tile);}
  }
}
function endStroke(){
  if(!state.stroke)return;const tiles=[];let bytes=0;
  for(const tile of state.stroke.values()){tile.after=copyTile(tile);if(tile.before.some((v,i)=>v!==tile.after[i])){tiles.push(tile);bytes+=tile.before.byteLength+tile.after.byteLength;}}
  state.stroke=null;if(!tiles.length)return;state.historyBytes-=state.redo.reduce((sum,item)=>sum+item.bytes,0);state.undo.push({tiles,bytes});state.historyBytes+=bytes;state.redo=[];
  while(state.historyBytes>100*1024*1024&&state.undo.length>1)state.historyBytes-=state.undo.shift().bytes;
  dirty();updateHistory();
}
function updateHistory(){$("undo").disabled=!state.undo.length;$("redo").disabled=!state.redo.length;}
function restore(redo=false){const from=redo?state.redo:state.undo,to=redo?state.undo:state.redo;if(!from.length)return;const change=from.pop();for(const tile of change.tiles){const data=redo?tile.after:tile.before;for(let y=0;y<tile.h;y++)state.ids.set(data.subarray(y*tile.w,(y+1)*tile.w),(tile.y+y)*state.width+tile.x);updatePixels(tile.x,tile.y,tile.x+tile.w,tile.y+tile.h);}to.push(change);dirty();updateHistory();}
function brush(a,b){
  const r=state.brush/2,x0=Math.max(0,Math.floor(Math.min(a.x,b.x)-r)),y0=Math.max(0,Math.floor(Math.min(a.y,b.y)-r)),x1=Math.min(state.width,Math.ceil(Math.max(a.x,b.x)+r)),y1=Math.min(state.height,Math.ceil(Math.max(a.y,b.y)+r));
  capture(x0,y0,x1,y1);const dx=b.x-a.x,dy=b.y-a.y,len=dx*dx+dy*dy;
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const t=len?Math.max(0,Math.min(1,((x+.5-a.x)*dx+(y+.5-a.y)*dy)/len)):0;const ex=x+.5-(a.x+t*dx),ey=y+.5-(a.y+t*dy);if(ex*ex+ey*ey<=r*r)state.ids[y*state.width+x]=state.label;}
  updatePixels(x0,y0,x1,y1);
}
function fillPolygon(vertices){
  if(vertices.length<3)return;beginStroke();const ys=vertices.map(p=>p.y),xs=vertices.map(p=>p.x),y0=Math.max(0,Math.floor(Math.min(...ys))),y1=Math.min(state.height,Math.ceil(Math.max(...ys))),x0=Math.max(0,Math.floor(Math.min(...xs))),x1=Math.min(state.width,Math.ceil(Math.max(...xs)));capture(x0,y0,x1,y1);
  for(let y=y0;y<y1;y++){const hits=[],scan=y+.5;for(let i=0,j=vertices.length-1;i<vertices.length;j=i++){const a=vertices[j],b=vertices[i];if((a.y>scan)!==(b.y>scan))hits.push(a.x+(scan-a.y)*(b.x-a.x)/(b.y-a.y));}hits.sort((a,b)=>a-b);for(let i=0;i+1<hits.length;i+=2){const start=Math.max(0,Math.ceil(hits[i]-.5)),end=Math.min(state.width,Math.ceil(hits[i+1]-.5));if(end>start)state.ids.fill(state.label,y*state.width+start,y*state.width+end);}}
  updatePixels(x0,y0,x1,y1);endStroke();state.polygon=[];requestDraw();
}
canvas.addEventListener("pointerdown",event=>{
  if(state.loading||!state.image)return;event.preventDefault();canvas.setPointerCapture(event.pointerId);const p=point(event);state.cursor=p;
  if(event.button===1||state.space||state.tool==="pan"){state.drag={type:"pan",x:event.clientX,y:event.clientY,tx:state.tx,ty:state.ty};return;}
  if(event.button!==0)return;if(event.altKey){const x=Math.floor(p.x),y=Math.floor(p.y);if(x>=0&&y>=0&&x<state.width&&y<state.height)selectLabel(state.ids[y*state.width+x]);return;}
  if(state.tool==="brush"){beginStroke();state.drag={type:"brush",last:p};brush(p,p);}
  else if(state.tool==="polygon"){if(event.detail<2)state.polygon.push(p);requestDraw();}
  else if(state.tool==="rectangle")state.drag={type:"rectangle",start:p};
});
canvas.addEventListener("pointermove",event=>{const p=point(event);state.cursor=p;if(state.drag?.type==="pan"){state.tx=state.drag.tx+event.clientX-state.drag.x;state.ty=state.drag.ty+event.clientY-state.drag.y;}else if(state.drag?.type==="brush"){brush(state.drag.last,p);state.drag.last=p;}requestDraw();});
function release(){if(state.drag?.type==="brush")endStroke();if(state.drag?.type==="rectangle"){const a=state.drag.start,b=state.cursor;fillPolygon([a,{x:b.x,y:a.y},b,{x:a.x,y:b.y}]);}state.drag=null;requestDraw();}
canvas.addEventListener("pointerup",release);canvas.addEventListener("pointercancel",()=>{endStroke();state.drag=null;requestDraw();});
canvas.addEventListener("dblclick",event=>{event.preventDefault();if(state.tool==="polygon")fillPolygon(state.polygon);});
canvas.addEventListener("pointerleave",()=>{if(!state.drag)state.cursor=null;requestDraw();});
canvas.addEventListener("contextmenu",event=>event.preventDefault());
canvas.addEventListener("wheel",event=>{event.preventDefault();const p=point(event),factor=Math.exp(-event.deltaY*.0015),next=Math.max(.02,Math.min(16,state.scale*factor));state.tx+=p.x*(state.scale-next);state.ty+=p.y*(state.scale-next);state.scale=next;requestDraw();},{passive:false});
async function fetchImage(url){const image=new Image();image.src=url;await image.decode();return image;}
async function localMask(entry){
  if(outputFolder&&entry.mask){try{const handle=await outputFolder.getFileHandle(entry.mask);return URL.createObjectURL(await handle.getFile());}catch(error){if(error.name!=="NotFoundError")throw error;}}
  return window.REFERENCE_BUNDLE.seeds[entry.id];
}
async function loadImage(id){
  if(state.loading)return;if(state.dirty&&!confirm("Discard unsaved changes and load another image?")){$("images").value=state.entry.id;return;}
  state.loading=true;document.body.classList.add("loading");status("Loading full-resolution image…");
  try{const entry=state.catalog.entries.find(e=>e.id===id);
    if(offline&&!localPhotos.has(entry.image)){status("Click Open photos and select images from data/examples");$("images").value=state.entry?.id||id;return;}
    let imageUrl,maskUrl;
    if(offline){imageUrl=URL.createObjectURL(localPhotos.get(entry.image));maskUrl=await localMask(entry);}else{imageUrl=`/api/image?id=${encodeURIComponent(id)}`;maskUrl=`/api/mask?id=${encodeURIComponent(id)}&revision=${entry.revision}`;}
    let image,mask;try{[image,mask]=await Promise.all([fetchImage(imageUrl),fetchImage(maskUrl)]);}finally{if(imageUrl.startsWith("blob:"))URL.revokeObjectURL(imageUrl);if(maskUrl.startsWith("blob:"))URL.revokeObjectURL(maskUrl);}
    state.image=image;state.entry=entry;state.width=image.naturalWidth;state.height=image.naturalHeight;
    if(mask.naturalWidth!==state.width||mask.naturalHeight!==state.height)throw Error("Image/mask size mismatch");
    labelCanvas.width=state.width;labelCanvas.height=state.height;labelCtx.drawImage(mask,0,0);const data=labelCtx.getImageData(0,0,state.width,state.height);state.ids=new Uint8Array(state.width*state.height);for(let i=0;i<state.ids.length;i++)state.ids[i]=data.data[i*4];state.rgba=data;
    updatePixels(0,0,state.width,state.height);state.undo=[];state.redo=[];state.historyBytes=0;state.polygon=[];state.drag=null;state.dirty=false;updateHistory();
    $("images").value=id;$("metadata").textContent=`${entry.width} × ${entry.height} · ${entry.status} · revision ${entry.revision}`;$("rule").textContent=entry.seed_rule;$("reviewNote").textContent=entry.review_note;$("notes").value=entry.notes;$("reviewed").checked=entry.status==="reviewed";fit();status("Ready · reference pending your review");
  }catch(error){status(error.message,true);}finally{state.loading=false;document.body.classList.remove("loading");}
}
function exportCanvas(){const out=document.createElement("canvas");out.width=state.width;out.height=state.height;const c=out.getContext("2d"),data=c.createImageData(state.width,state.height);for(let i=0;i<state.ids.length;i++){const q=i*4;data.data[q]=state.ids[i];data.data[q+1]=state.ids[i];data.data[q+2]=state.ids[i];data.data[q+3]=255;}c.putImageData(data,0,0);return out;}
async function writeLocal(name,data){const handle=await outputFolder.getFileHandle(name,{create:true});const stream=await handle.createWritable();await stream.write(data);await stream.close();}
async function sha256(blob){const hash=await crypto.subtle.digest("SHA-256",await blob.arrayBuffer());return Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,"0")).join("");}
async function saveOffline(){
  if(!outputFolder)throw Error("Choose Output folder first (data/reference_masks/review_v1)");
  const photo=localPhotos.get(state.entry.image);if(await sha256(photo)!==state.entry.image_sha256)throw Error("The selected photo differs from the reference source");
  try{const file=await (await outputFolder.getFileHandle("manifest.json")).getFile();const manifest=JSON.parse(await file.text()),existing=manifest.entries.find(e=>e.id===state.entry.id);if(existing&&existing.revision!==state.entry.revision)throw Error("Mask changed on disk. Reselect the output folder before saving.");}catch(error){if(error.name!=="NotFoundError")throw error;}
  const blob=await new Promise(resolve=>exportCanvas().toBlob(resolve,"image/png")),revision=state.entry.revision+1;
  const filename=`${state.entry.id}.edit_${String(revision).padStart(3,"0")}.png`;
  try{await outputFolder.getFileHandle(filename);throw Error("This revision already exists. Reselect the output folder.");}catch(error){if(error.name!=="NotFoundError")throw error;}
  const updated={...state.entry,mask:filename,mask_sha256:await sha256(blob),revision,status:$("reviewed").checked?"reviewed":"manual_draft",notes:$("notes").value,saved_at:new Date().toISOString()};
  await writeLocal(filename,blob);
  const manifest={...state.catalog,entries:state.catalog.entries.map(e=>e.id===updated.id?updated:e)};
  await writeLocal("manifest.json",JSON.stringify(manifest,null,2)+"\n");state.catalog=manifest;
  return {entry:updated};
}
async function save(){
  if(!state.image||state.loading)return;if(state.drag)release();if(state.polygon.length){status("Finish the polygon with Enter, or cancel with Esc before saving",true);return;}
  state.loading=true;document.body.classList.add("loading");status("Saving full-resolution class-ID mask…");
  try{let result;if(offline){result=await saveOffline();}else{const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:state.entry.id,revision:state.entry.revision,mask_png:exportCanvas().toDataURL("image/png"),notes:$("notes").value,reviewed:$("reviewed").checked})});result=await response.json();if(!response.ok)throw Error(result.error);}Object.assign(state.entry,result.entry);state.dirty=false;$("metadata").textContent=`${state.width} × ${state.height} · ${state.entry.status} · revision ${state.entry.revision}`;status(`Saved locally · ${state.entry.status} · revision ${state.entry.revision}`);}catch(error){status(error.message,true);}finally{state.loading=false;document.body.classList.remove("loading");}
}
$("save").onclick=save;$("download").onclick=()=>{if(!state.image)return;exportCanvas().toBlob(blob=>{const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=state.entry.id+".mask.png";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});};
$("images").onchange=()=>loadImage($("images").value);
function navigate(step){if(!state.entry)return;const entries=offline?state.catalog.entries.filter(e=>localPhotos.has(e.image)):state.catalog.entries;const i=entries.findIndex(e=>e.id===state.entry.id);loadImage(entries[(i+step+entries.length)%entries.length].id);}
$("previous").onclick=()=>navigate(-1);$("next").onclick=()=>navigate(1);
$("undo").onclick=()=>restore();$("redo").onclick=()=>restore(true);$("fit").onclick=fit;$("actual").onclick=()=>{state.scale=1;state.tx=(canvas.width-state.width)/2;state.ty=(canvas.height-state.height)/2;requestDraw();};
function setBrush(value){state.brush=Math.max(2,Math.min(400,Math.round(value)));$("brushSize").value=state.brush;$("brushValue").textContent=state.brush+" px";requestDraw();}
$("brushSize").oninput=()=>setBrush(Number($("brushSize").value));$("smaller").onclick=()=>setBrush(state.brush/1.4);$("larger").onclick=()=>setBrush(state.brush*1.4);
$("opacity").oninput=()=>{state.opacity=Number($("opacity").value)/100;$("opacityValue").textContent=$("opacity").value+"%";requestDraw();};$("guides").onchange=requestDraw;
$("notes").oninput=()=>{state.dirty=true;status("Unsaved notes");};$("reviewed").onchange=()=>{state.dirty=true;status("Review status changed · save to apply");};
window.addEventListener("keydown",event=>{
  if(state.loading)return;
  if(["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName)&&!event.ctrlKey&&!event.metaKey)return;
  const key=event.key.toLowerCase(),control=event.ctrlKey||event.metaKey;
  if(control&&key==="s"){event.preventDefault();save();return;}if(control&&key==="z"){event.preventDefault();restore(event.shiftKey);return;}
  if(event.code==="Space"){event.preventDefault();state.space=true;canvas.style.cursor="grab";}else if(key==="h"){state.hide=true;requestDraw();}else if(key==="b")selectTool("brush");else if(key==="p")selectTool("polygon");else if(key==="r")selectTool("rectangle");else if(key==="f")fit();else if(key==="i")selectLabel(255);else if("01234".includes(key)&&key.length===1)selectLabel(Number(key));else if(key==="[")setBrush(state.brush/1.4);else if(key==="]")setBrush(state.brush*1.4);else if(key==="enter")fillPolygon(state.polygon);else if(key==="escape"){state.polygon=[];requestDraw();}
});
window.addEventListener("keyup",event=>{if(event.code==="Space"){state.space=false;canvas.style.cursor=state.tool==="pan"?"grab":"crosshair";}if(event.key.toLowerCase()==="h"){state.hide=false;requestDraw();}});
window.addEventListener("blur",()=>{state.space=false;state.hide=false;if(state.drag)release();requestDraw();});
window.addEventListener("beforeunload",event=>{if(state.dirty){event.preventDefault();event.returnValue="";}});
$("openPhotos").onclick=()=>$("photoFiles").click();
$("photoFiles").onchange=()=>{for(const file of $("photoFiles").files)localPhotos.set(file.name,file);const first=state.catalog.entries.find(e=>localPhotos.has(e.image));if(first)loadImage(first.id);else status("Select the matching original photos from data/examples",true);};
$("outputFolder").onclick=async()=>{
  if(!window.showDirectoryPicker){status("Use Edge or Chrome for folder saving; PNG download is also available",true);return;}
  try{const selected=await window.showDirectoryPicker({id:"semseg-reference-masks",mode:"readwrite"});let stored;
    try{stored=JSON.parse(await (await (await selected.getFileHandle("manifest.json")).getFile()).text());if(stored.schema_version!==1||!Array.isArray(stored.entries)||stored.entries.length!==state.catalog.entries.length||stored.entries.some(e=>!state.catalog.entries.some(current=>current.id===e.id&&current.image_sha256===e.image_sha256)))throw Error("This folder contains a different annotation manifest");if(state.dirty){const existing=stored.entries.find(e=>e.id===state.entry.id);if(existing.revision!==state.entry.revision||existing.mask_sha256!==state.entry.mask_sha256)throw Error("This folder has a different version. Download the current mask before switching.");}}catch(error){if(error.name!=="NotFoundError")throw error;}
    if(stored){for(const entry of stored.entries){if(!entry.mask&&entry.revision===0){const proposal=window.REFERENCE_BUNDLE.catalog.entries.find(e=>e.id===entry.id);for(const field of ["seed","seed_sha256","seed_rule","review_note"])entry[field]=proposal[field];}}}
    outputFolder=selected;if(stored)state.catalog=stored;$("outputFolder").textContent="Folder: "+selected.name;
    if(state.entry&&!state.dirty){await loadImage(state.entry.id);}status(state.dirty?"Output folder ready · your unsaved edits are preserved":"Output folder ready · saves create new mask revisions");
  }catch(error){if(error.name!=="AbortError")status(error.message,true);}
};
async function init(){try{if(offline){state.catalog=window.REFERENCE_BUNDLE.catalog;$("openPhotos").hidden=false;$("outputFolder").hidden=false;document.body.classList.add("offline");}else{const response=await fetch("/api/catalog");state.catalog=await response.json();}for(const entry of state.catalog.entries){const option=document.createElement("option");option.value=entry.id;option.textContent=`${String(entry.number).padStart(2,"0")} · ${entry.image}`;$("images").appendChild(option);}await loadImage(state.catalog.entries[0].id);}catch(error){status(error.message,true);}}
init();
