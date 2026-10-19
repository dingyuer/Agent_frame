from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .service import router, workbench
from .config import ROOT

@asynccontextmanager
async def lifespan(app):
    yield
    await workbench.close()

app=FastAPI(title='Jev Decision Lab',version='1.0.0',lifespan=lifespan)

@app.middleware('http')
async def local_origin(request: Request, call_next):
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
        return JSONResponse({'detail':'请通过本服务页面发起请求'},status_code=403)
    return await call_next(request)

app.include_router(router)
STATIC=ROOT/'viz'/'static'
app.mount('/static',StaticFiles(directory=STATIC),name='static')

@app.get('/')
@app.get('/jev-compare')
@app.get('/jev/compare')
def index(): return FileResponse(STATIC/'jev-compare.html')
