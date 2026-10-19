import threading
from fastapi.testclient import TestClient
from osworld_agent.viz import server


def test_chat_result_reports_usage_and_model_without_credentials(monkeypatch):
    import requests
    seen = []
    class Response:
        status_code = 200
        def json(self):
            return {'model':'actual-model','usage':{'prompt_tokens':12,'completion_tokens':3},
                    'choices':[{'message':{'content':'result'}}]}
    monkeypatch.setattr(requests,'post',lambda *a,**kw:seen.append(kw) or Response())
    result = TestClient(server.app).post('/debug/chat',json={
        'engine':{'model':'requested','base_url':'http://test.invalid/v1'},
        'messages':[{'role':'user','content':'hello'}],'api_key':'private-test-key','call_type':'decision'}).json()
    assert result['ok'] and result['model']=='actual-model'
    assert result['usage']['prompt_tokens']==12 and result['latency_ms']>=0
    assert 'private-test-key' not in str(result)
    assert seen[0]['headers']['Authorization']=='Bearer private-test-key'


def test_debug_request_runs_off_event_loop(monkeypatch):
    called=[]
    monkeypatch.setattr(server,'_debug_chat_request',lambda payload:called.append(threading.get_ident()) or {'ok':True})
    with TestClient(server.app) as client:
        loop_thread=client.portal.call(threading.get_ident)
        assert client.post('/debug/chat',json={}).json()['ok']
    assert called and called[0]!=loop_thread


def test_decision_text_does_not_get_grounding_annotation(monkeypatch):
    import requests
    class Response:
        status_code=200
        def json(self):return {'choices':[{'message':{'content':'(100, 200)'}}]}
    monkeypatch.setattr(requests,'post',lambda *a,**kw:Response())
    monkeypatch.setattr(server,'_make_annotation_image',lambda *a,**kw:'data:image/png;base64,test')
    body={'engine':{'model':'test','base_url':'http://test.invalid/v1'},'messages':[{'role':'user','content':'where?'}],'call_type':'decision'}
    result=server._debug_chat_request(body)
    assert result['annotation_image'] is None
    result=server._debug_chat_request({**body,'call_type':'grounding'})
    assert result['annotation_image']=='data:image/png;base64,test'


def test_debug_transport_does_not_inherit_socks_proxy(monkeypatch):
    import requests
    seen=[]
    class Response:
        status_code=200
        def json(self):return {'choices':[{'message':{'content':'ok'}}]}
    monkeypatch.setenv('ALL_PROXY','socks5h://unreachable.invalid:1080')
    monkeypatch.delenv('AGENT_DEBUG_HTTP_PROXY',raising=False)
    monkeypatch.setenv('AGENT_KNOWLEDGE_HTTP_PROXY','http://egress.invalid:8080')
    monkeypatch.setattr(requests,'post',lambda *a,**kw:seen.append(kw['proxies']) or Response())
    body={'engine':{'model':'test','base_url':'http://10.0.0.1/v1'},'messages':[{'role':'user','content':'hi'}]}
    assert server._debug_chat_request(body)['ok']
    assert seen[-1]=={'http':'','https':'','all':''}
    body['engine']['base_url']='https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert server._debug_chat_request(body)['ok']
    assert seen[-1]=={'http':'http://egress.invalid:8080','https':'http://egress.invalid:8080','all':''}
