import json
import pytest
from fastapi.testclient import TestClient

from osworld_agent.knowledge_base import load_knowledge, save_knowledge, retrieve_knowledge, validate_document


def entry(ident='campus', **extra):
    return {'id': ident, 'title': '校园门户', 'aliases': ['校园官网', '学校官网'],
            'keywords': ['校园', '通知'], 'url': 'https://campus.example.test',
            'content': '通知位于公告栏目。', 'enabled': True, **extra}


def snapshot(*entries):
    return {'revision': 'v1', 'document': validate_document({'version': 1, 'entries': list(entries)})}


def test_alias_keyword_and_acronym_retrieval_exclude_unrelated_or_disabled():
    doc = snapshot(entry(), entry('rfq', title='采购系统', aliases=['RFQ'], keywords=[], url='https://rfq.test'),
                   entry('off', title='停用系统', aliases=['停用系统'], enabled=False))
    assert [e['id'] for e in retrieve_knowledge('去学校官网看新闻', doc)['entries']] == ['campus']
    assert [e['id'] for e in retrieve_knowledge('查校园通知', doc)['entries']] == ['campus']
    assert not retrieve_knowledge('查校园', doc)['entries']
    assert not retrieve_knowledge('查看abcRFQxyz，停用系统', doc)['entries']
    assert [e['id'] for e in retrieve_knowledge('查看 RFQ 状态', doc)['entries']] == ['rfq']


def test_large_knowledge_is_bounded_and_ambiguity_explicit():
    doc = snapshot(*(entry(str(i), url=f'https://campus{i}.test', content='说明'*1000) for i in range(1000)))
    found = retrieve_knowledge('访问校园官网', doc)
    assert found['matched_count'] == 1000
    assert len(found['entries']) <= 4 and found['selected_chars'] <= 6000
    assert '校园官网' in found['ambiguous_terms']
    assert all(len(e['content']) == 2000 for e in found['entries'])  # no partial restrictions


def test_roundtrip_preserves_unicode_and_rejects_stale_save(tmp_path):
    path = tmp_path/'knowledge.yaml'
    first = load_knowledge(path)
    saved = save_knowledge({'version':1, 'entries':[entry()]}, first['revision'], path)
    assert load_knowledge(path) == saved
    assert '校园门户' in path.read_text()
    with pytest.raises(FileExistsError):
        save_knowledge({'version':1,'entries':[]}, first['revision'], path)
    assert len(load_knowledge(path)['document']['entries']) == 1


@pytest.mark.parametrize('change', [{'url':'javascript:alert(1)'}, {'url':'https://user:pass@site.test'},
                                    {'aliases':'campus'}, {'enabled':'true'}, {'content':'x'*3001}])
def test_invalid_knowledge_is_rejected(change):
    with pytest.raises(ValueError):
        snapshot(entry(**change))


def test_editor_api_preview_does_not_save_and_conflicting_edit_is_rejected(monkeypatch,tmp_path):
    monkeypatch.setenv('AGENT_KNOWLEDGE_PATH',str(tmp_path/'knowledge.yaml'))
    from osworld_agent.viz.server import app
    client=TestClient(app)
    loaded=client.get('/knowledge').json()
    doc={'version':1,'entries':[entry()]}
    preview=client.post('/knowledge/preview',json={'task':'校园官网','document':doc}).json()
    assert preview['entries'][0]['id']=='campus'
    assert client.get('/knowledge').json()['document']['entries']==[]
    assert client.put('/knowledge',json={'document':doc,'revision':loaded['revision']}).status_code==200
    assert client.put('/knowledge',json={'document':doc,'revision':loaded['revision']}).status_code==409
    assert client.post('/knowledge/preview',json={'task':''}).status_code==422
