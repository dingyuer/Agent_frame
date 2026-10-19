"""Comparable inputs; explicit protocol validation and separate abstention policy."""
import asyncio
import json
import math
import os
import time
from urllib.parse import urlsplit
import httpx
from .fixtures import evidence, evidence_hash

PROTOCOL_VERSION = 3

def labels_for(q):
    if q['type']=='noul': return q.get('criteria', {'false':'否','true':'是'})
    if q['type']=='score': return {str(i):v for i,v in enumerate(q['criteria'])}
    return q['criteria']


def make_payload(c, provider, settings, mode='json'):
    shared=evidence(c)
    if provider=='jev': return {'model':settings.jev_model,**shared}
    contract={k:list(labels_for(q)) for k,q in c['questions'].items()}
    instructions='请仅根据给定截图和状态证据，逐项回答下列判断问题。两种模型接收到的状态和问题含义相同。'
    if mode=='json':
        shape={'answers':{k:'<'+ '|'.join(v)+'>' for k,v in contract.items()}}
        instructions+='只返回一个 JSON 对象，字段结构必须为：'+json.dumps(shape,ensure_ascii=False)+'。将每个尖括号内的占位值替换为该问题允许的一个字符串标签。保持问题 ID 原样，每题恰好回答一次，不要创建通用的 question_id 字段。允许的标签为：'+json.dumps(contract,ensure_ascii=False)+'。score 类型请选择最合适的离散等级编号，以字符串返回，不要返回小数期望值。JSON 以外不附加解释或 Markdown 标记。'
    else:
        instructions+='请用中文逐项说明你的选择和简短理由，同时注明对应问题 ID 与所选标签。仅在给定选项中选择，不要生成其他标签。允许的标签为：'+json.dumps(contract,ensure_ascii=False)
    text=instructions+'\n\n【任务与状态证据】\n'+shared['state']+'\n\n【判断问题与选项含义】\n'+json.dumps(shared['questions'],ensure_ascii=False)
    payload={'model':settings.vlm_model,'messages':[{'role':'user','content':[{'type':'text','text':text}]+[{'type':'image_url','image_url':{'url':x}} for x in shared['images']]}], 'temperature':0,'max_tokens':768}
    if settings.vlm_thinking_mode=='dashscope':
        payload['enable_thinking']=settings.vlm_enable_thinking
    elif settings.vlm_thinking_mode=='chat_template':
        payload['chat_template_kwargs']={'enable_thinking':settings.vlm_enable_thinking}
    if mode=='json' and settings.vlm_json_mode:
        payload['response_format']={'type':'json_object'}
    return payload


def number(x, low, high):
    return type(x) in (float,int) and math.isfinite(x) and low <= x <= high


def parse_jev(body, questions, threshold, margin):
    if not isinstance(body,dict) or not isinstance(body.get('answers'),dict):
        raise ValueError('Missing answers object')
    if set(body['answers']) != set(questions):
        raise ValueError('Question IDs differ from request')
    result={}
    for key,q in questions.items():
        ans=body['answers'][key]
        if not isinstance(ans,dict) or ans.get('error') or ans.get('type')!=q['type']:
            raise ValueError('Invalid typed answer: '+key)
        labels=labels_for(q)
        if q['type']=='noul':
            p=ans.get('noul')
            if not number(p,0,1): raise ValueError('Invalid noul probability')
            probs={'false':1-p,'true':p}
        else:
            probs=ans.get('probabilities')
            if not isinstance(probs,dict) or set(probs)!=set(labels) or any(not number(p,0,1) for p in probs.values()) or abs(sum(probs.values())-1)>.001:
                raise ValueError('Invalid probability distribution: '+key)
        ranked=sorted(probs,key=probs.get,reverse=True)
        top=ranked[0]; gap=probs[top]-(probs[ranked[1]] if len(ranked)>1 else 0)
        if q['type']=='choice' and 'choice' in ans and (ans['choice'] not in probs or abs(probs[ans['choice']]-probs[top])>1e-9):
            raise ValueError('Choice differs from distribution')
        expected_score=None
        if q['type']=='score':
            expected_score=sum(int(k)*v for k,v in probs.items())
            if not number(ans.get('score'),0,len(labels)-1) or abs(ans['score']-expected_score)>.001 or ans.get('legend')!=labels:
                raise ValueError('Score or legend differs from distribution')
        abstained=probs[top]<threshold or gap<margin
        result[key]={'type':q['type'],'label':top,'decision_label':'unknown' if abstained else top,'abstained':abstained,
                     'probabilities':probs,'top_probability':probs[top],'margin':gap,'score':expected_score}
    return result


def parse_vlm(text, questions):
    parsed=json.loads(text)
    if not isinstance(parsed,dict) or set(parsed)!= {'answers'} or not isinstance(parsed['answers'],dict) or set(parsed['answers'])!=set(questions):
        raise ValueError('Expected exact answers object')
    out={}
    for key,q in questions.items():
        label=parsed['answers'][key]
        # Require strings as requested; no silent repair or second inference call.
        if not isinstance(label,str) or label not in labels_for(q):
            raise ValueError('Label outside schema: '+key)
        out[key]={'type':q['type'],'label':label,'decision_label':label,'abstained':False}
    return out


def score_answers(answers, expected):
    return {'correct':sum(answers.get(k,{}).get('label')==v for k,v in expected.items()),'total':len(expected),
            'accepted':sum(not a.get('abstained',False) for a in answers.values()),
            'accepted_correct':sum(not answers.get(k,{}).get('abstained',True) and answers.get(k,{}).get('label')==v for k,v in expected.items())}


def clean_secret(value, settings):
    def finite(item):
        if isinstance(item,float) and not math.isfinite(item): return '[non-finite number]'
        if isinstance(item,dict): return {k:finite(v) for k,v in item.items()}
        if isinstance(item,list): return [finite(v) for v in item]
        return item
    text=json.dumps(finite(value),ensure_ascii=False,default=str,allow_nan=False)
    for secret in (settings.jev_api_key,settings.vlm_api_key):
        if secret: text=text.replace(secret,'[redacted]')
    return json.loads(text)


class ModelClients:
    def __init__(self): self.clients={}
    def client(self,url,trust_env):
        # Explicit http(s) proxy selection avoids requiring SOCKS support for
        # an unrelated ALL_PROXY when a machine already has HTTPS_PROXY.
        scheme=urlsplit(url).scheme
        proxy=(os.environ.get(scheme+'_proxy') or os.environ.get(scheme.upper()+'_PROXY') or os.environ.get('ALL_PROXY') or os.environ.get('all_proxy')) if trust_env else None
        if trust_env:
            no_proxy=os.environ.get('NO_PROXY',os.environ.get('no_proxy',''))
            host=urlsplit(url).hostname or ''
            if host in ('localhost','127.0.0.1','::1') or any(x.strip()=='*' or host==x.strip().lstrip('.') or host.endswith('.'+x.strip().lstrip('.')) for x in no_proxy.split(',') if x.strip()): proxy=None
        key=(proxy,scheme)
        if key not in self.clients:
            self.clients[key]=httpx.AsyncClient(trust_env=False,proxy=proxy,limits=httpx.Limits(max_connections=8,max_keepalive_connections=4))
        return self.clients[key]
    async def close(self):
        for client in self.clients.values(): await client.aclose()
        self.clients.clear()

    async def run(self,c,provider,settings,mode='json'):
        url=getattr(settings,provider+'_url'); requested=getattr(settings,provider+'_model')
        result={'provider':provider,'requested_model':requested,'model':requested,'mode':'typed' if provider=='jev' else mode,
                'evidence_hash':evidence_hash(c),'status':'error','answers':{},'latency_ms':None,
                'score':{'correct':0,'total':len(c['expected']),'accepted':0,'accepted_correct':0},'usage':{},
                'policy':{'threshold':settings.threshold,'margin':settings.margin} if provider=='jev' else None,
                'transport':{'endpoint':url,'proxy_enabled':getattr(settings,provider+'_trust_env')},
                'generation':{'temperature':0,'max_tokens':768,'enable_thinking':settings.vlm_enable_thinking,'json_mode':mode=='json' and settings.vlm_json_mode} if provider=='vlm' else None}
        if not url:
            return {**result,'error':'尚未配置模型地址，请在连接设置中填写。','error_type':'not_configured'}
        payload=make_payload(c,provider,settings,mode)
        # Archive the exact textual request, referencing bundled image bytes.
        archived=json.loads(json.dumps(payload))
        if provider=='jev':
            archived['images']=['asset:'+name+'.png' for name in c['assets']]
        else:
            for part,name in zip(archived['messages'][0]['content'][1:],c['assets']):
                part['image_url']['url']='asset:'+name+'.png'
        result['request']=archived
        result['protocol_version']=PROTOCOL_VERSION
        key=getattr(settings,provider+'_api_key'); headers={'Authorization':'Bearer '+key} if key else {}
        started=time.perf_counter()
        try:
            client=self.client(url,getattr(settings,provider+'_trust_env'))
            response=await asyncio.wait_for(client.post(url,json=payload,headers=headers,timeout=httpx.Timeout(settings.timeout,connect=min(10,settings.timeout))),timeout=settings.timeout+1)
            result['latency_ms']=round((time.perf_counter()-started)*1000,2)
            if response.status_code>=400:
                result.update(error=f'模型服务返回 HTTP {response.status_code}',error_type='http_error',http_status=response.status_code)
                return result
            body=response.json()
            if not isinstance(body,dict): raise ValueError('Response must be an object')
            result.update(raw_response=clean_secret(body,settings),model=body.get('model',requested),usage=body.get('usage',{}),model_version=body.get('model_version'))
            if provider=='jev':
                result['answers']=parse_jev(body,c['questions'],settings.threshold,settings.margin)
            else:
                msg=body['choices'][0]['message']; text=msg.get('content')
                if not isinstance(text,str): raise ValueError('Missing text content')
                result['text']=text
                if mode=='free':
                    result.update(status='text',score=None,format_valid=None)
                    return clean_secret(result,settings)
                result['answers']=parse_vlm(text,c['questions'])
            result.update(status='ok',format_valid=True,score=score_answers(result['answers'],c['expected']))
        except (httpx.TimeoutException, asyncio.TimeoutError):
            result.update(error='调用超时；本次保留为失败记录，没有自动重试。',error_type='timeout')
        except httpx.RequestError:
            result.update(error='无法连接模型服务，请检查地址、端口和代理设置。',error_type='connection')
        except (ValueError,KeyError,TypeError,IndexError):
            result.update(error='响应不符合请求的标签或概率契约；原始响应可在下方检查。',error_type='invalid_response',format_valid=False)
        except Exception as exc:
            result.update(error='模型调用未完成：'+type(exc).__name__,error_type='client_error')
        if result['latency_ms'] is None: result['latency_ms']=round((time.perf_counter()-started)*1000,2)
        return clean_secret(result,settings)
