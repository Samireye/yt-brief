"""
YT Brief: paste a YouTube link, get the transcript, then have Claude analyze it
against your own profile.

Run (once):   pip install fastapi uvicorn youtube-transcript-api anthropic
Then:         export ANTHROPIC_API_KEY=sk-ant-...        (Windows: set ANTHROPIC_API_KEY=...)
              python app.py
Open:         http://127.0.0.1:8000

Optional: ANTHROPIC_MODEL to override the model (default claude-sonnet-5-5).
"""
import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_CHARS = 200_000  # transcript characters sent to the model

app = FastAPI(title="YT Brief")

# Password gate (HTTP Basic). Set APP_PASSWORD; username can be anything.
# On Vercel it is required, so nobody else can spend your API credits.
import base64
import hmac

from fastapi import Request
from fastapi.responses import Response

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
REQUIRE_PW = bool(os.environ.get("VERCEL")) or bool(APP_PASSWORD)


@app.middleware("http")
async def gate(request: Request, call_next):
    if not REQUIRE_PW:
        return await call_next(request)
    if not APP_PASSWORD:
        return Response("Set APP_PASSWORD in the environment.", status_code=500)
    ok = False
    h = request.headers.get("authorization", "")
    if h.lower().startswith("basic "):
        try:
            pw = base64.b64decode(h[6:]).decode().split(":", 1)[1]
            ok = hmac.compare_digest(pw, APP_PASSWORD)
        except Exception:
            ok = False
    if not ok:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="YT Brief"'})
    return await call_next(request)

ID_PATTERNS = [
    r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
    r"(?:v=)([A-Za-z0-9_-]{11})",
    r"(?:shorts/|embed/|live/)([A-Za-z0-9_-]{11})",
]


def video_id(url: str) -> str:
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    for p in ID_PATTERNS:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Could not find a YouTube video id in that link.")


def fetch_via_supadata(vid: str, timestamps: bool) -> str:
    """Hosted transcript API (not blocked on cloud hosts). Needs SUPADATA_API_KEY."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    q = urllib.parse.urlencode(
        {"url": f"https://www.youtube.com/watch?v={vid}", "mode": "native", "text": "false"}
    )
    req = urllib.request.Request(
        f"https://api.supadata.ai/v1/transcript?{q}",
        headers={"x-api-key": os.environ["SUPADATA_API_KEY"]},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            status, data = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Supadata error {e.code}: {e.read()[:200].decode(errors='ignore')}")
    if status != 200 or not isinstance(data.get("content"), list):
        raise RuntimeError("No transcript available for this video.")
    lines = []
    for c in data["content"]:
        t = " ".join(str(c.get("text", "")).split())
        if not t:
            continue
        if timestamps:
            m, sec = divmod(int(c.get("offset", 0)) // 1000, 60)
            lines.append(f"[{m:02d}:{sec:02d}] {t}")
        else:
            lines.append(t)
    return "\n".join(lines) if timestamps else " ".join(lines)


def fetch_transcript(vid: str, timestamps: bool = False) -> str:
    """Return the transcript as text. Uses Supadata if configured, else youtube-transcript-api."""
    if os.environ.get("SUPADATA_API_KEY"):
        return fetch_via_supadata(vid, timestamps)
    from youtube_transcript_api import YouTubeTranscriptApi

    try:  # v1.x
        fetched = YouTubeTranscriptApi().fetch(vid, languages=["en", "en-US", "en-GB"])
        rows = [(s.start, s.text) for s in fetched]
    except AttributeError:  # v0.x
        data = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
        rows = [(d["start"], d["text"]) for d in data]

    lines = []
    for start, text in rows:
        text = " ".join(text.split())
        if not text:
            continue
        if timestamps:
            m, s = divmod(int(start), 60)
            lines.append(f"[{m:02d}:{s:02d}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines) if timestamps else " ".join(lines)


MODES = {
    "summary": (
        "Summarize this video. Give: a 3-sentence overview, the key points as a short list, "
        "and anything that is sponsored or promotional (note it plainly)."
    ),
    "critique": (
        "Evaluate the claims in this video. List the main claims, then for each say whether it is "
        "well supported, plausible but unverified, or hype/misleading, and why. Flag false precision, "
        "internal contradictions, and sales funnels. Be direct and concise."
    ),
    "relevance": (
        "Using the profile below, explain how this video's ideas apply to this person specifically: "
        "what is useful, what to ignore, what it would change about their plan, and 2-3 concrete next "
        "steps. If something conflicts with the profile, say so."
    ),
}

SYSTEM = (
    "You analyze YouTube transcripts for one person. Be accurate and specific. Do not invent facts "
    "that are not in the transcript. If you are not sure about a current-events claim, say you "
    "cannot verify it. Do not give personalized financial advice as certainty; frame it as "
    "information. Use short Markdown headings and bullets."
)


def analyze(transcript: str, mode: str, profile: str, question: str) -> str:
    import anthropic

    if mode not in MODES:
        raise ValueError("Unknown mode.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set on the server.")
    client = anthropic.Anthropic()
    task = MODES[mode]
    if question.strip():
        task += f"\n\nAlso answer this question: {question.strip()}"
    user = f"{task}\n\n"
    if profile.strip():
        user += f"<profile>\n{profile.strip()}\n</profile>\n\n"
    user += f"<transcript>\n{transcript[:MAX_CHARS]}\n</transcript>"
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class TranscriptReq(BaseModel):
    url: str
    timestamps: bool = False


class AnalyzeReq(BaseModel):
    transcript: str
    mode: str = "summary"
    profile: str = ""
    question: str = ""


@app.post("/api/transcript")
def api_transcript(req: TranscriptReq):
    try:
        vid = video_id(req.url)
        text = fetch_transcript(vid, req.timestamps)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # transcripts disabled, blocked IP, no captions
        raise HTTPException(
            502,
            f"Could not fetch a transcript ({type(e).__name__}: {str(e)[:150]}). The video may have captions turned "
            "off, or YouTube may be blocking this network. You can paste a transcript manually.",
        )
    return {"id": vid, "transcript": text, "chars": len(text)}


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq):
    if not req.transcript.strip():
        raise HTTPException(400, "No transcript to analyze.")
    try:
        return {"analysis": analyze(req.transcript, req.mode, req.profile, req.question)}
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"The AI request failed ({type(e).__name__}): {e}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YT Brief</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#171b24;--muted:#5d6678;--line:#dde1e8;--accent:#2f56d6}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b23;--fg:#e8ebf1;--muted:#98a1b3;--line:#272d38;--accent:#8ba3ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;padding:20px 16px 48px}
main{max-width:860px;margin:0 auto;display:grid;gap:16px}
h1{margin:0;font-size:26px}p.s{margin:4px 0 0;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;display:grid;gap:10px}
label{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
input[type=text],textarea,select{width:100%;font:inherit;padding:9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
textarea{min-height:110px;resize:vertical}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:9px 14px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
button.alt{background:transparent;color:var(--accent)}button:disabled{opacity:.5;cursor:wait}
#out{line-height:1.6}#out h1,#out h2,#out h3{margin:14px 0 4px}#out ul{margin:4px 0 4px 20px;padding:0}
.status{color:var(--muted);font-size:13px}.err{color:#c0392b}
</style></head><body><main>
<header><h1>YT Brief</h1><p class="s">Paste a YouTube link, pull the transcript, and have Claude read it against your profile.</p></header>

<section class="card">
  <label for="url">YouTube link</label>
  <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=...">
  <div class="row">
    <button id="get">Get transcript</button>
    <label style="text-transform:none;letter-spacing:0"><input type="checkbox" id="ts"> include timestamps</label>
    <span class="status" id="tstatus"></span>
  </div>
  <label for="tr">Transcript (you can also paste one here)</label>
  <textarea id="tr" placeholder="Transcript appears here"></textarea>
  <div class="row"><button class="alt" id="copy">Copy</button><button class="alt" id="dl">Download .txt</button></div>
</section>

<section class="card">
  <label for="profile">Your profile (saved in this browser, sent only with Analyze)</label>
  <textarea id="profile" placeholder="Example: 32, works in consulting, saving for a house, investing $X/month in index funds, wants to retire early. Edit this."></textarea>
  <div class="row">
    <div style="flex:1;min-width:200px"><label for="mode">What to do</label>
      <select id="mode"><option value="summary">Summarize</option><option value="critique">Check the claims</option><option value="relevance" selected>How it applies to me</option></select></div>
    <div style="flex:2;min-width:220px"><label for="q">Optional question</label><input type="text" id="q" placeholder="e.g. Is the bottom-of-market date reliable?"></div>
  </div>
  <div class="row"><button id="go">Analyze</button><span class="status" id="astatus"></span></div>
</section>

<section class="card"><label>Result</label><div id="out"></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
try{$('profile').value=localStorage.getItem('ytb_profile')||''}catch(e){}
$('profile').addEventListener('input',()=>{try{localStorage.setItem('ytb_profile',$('profile').value)}catch(e){}});
async function post(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(j.detail||('Request failed ('+r.status+')'));
  return j;
}
function md(s){
  const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  let h='',inList=false;
  for(const raw of s.split('\n')){
    let l=esc(raw);
    l=l.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
    const hm=l.match(/^(#{1,3})\s+(.*)/), bm=l.match(/^\s*[-*]\s+(.*)/);
    if(bm){ if(!inList){h+='<ul>';inList=true} h+='<li>'+bm[1]+'</li>'; continue }
    if(inList){h+='</ul>';inList=false}
    if(hm){ h+='<h'+hm[1].length+'>'+hm[2]+'</h'+hm[1].length+'>' }
    else if(l.trim()){ h+='<p>'+l+'</p>' }
  }
  if(inList) h+='</ul>';
  return h;
}
$('get').onclick=async()=>{
  $('tstatus').className='status';$('tstatus').textContent='Fetching...';$('get').disabled=true;
  try{const j=await post('/api/transcript',{url:$('url').value,timestamps:$('ts').checked});
    $('tr').value=j.transcript;$('tstatus').textContent=j.chars.toLocaleString()+' characters'}
  catch(e){$('tstatus').className='status err';$('tstatus').textContent=e.message}
  $('get').disabled=false;
};
$('go').onclick=async()=>{
  $('astatus').className='status';$('astatus').textContent='Thinking...';$('go').disabled=true;
  try{const j=await post('/api/analyze',{transcript:$('tr').value,mode:$('mode').value,profile:$('profile').value,question:$('q').value});
    $('out').innerHTML=md(j.analysis);$('astatus').textContent=''}
  catch(e){$('astatus').className='status err';$('astatus').textContent=e.message}
  $('go').disabled=false;
};
$('copy').onclick=async()=>{try{await navigator.clipboard.writeText($('tr').value)}catch(e){$('tr').select()}};
$('dl').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([$('tr').value],{type:'text/plain'}));a.download='transcript.txt';a.click()};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
