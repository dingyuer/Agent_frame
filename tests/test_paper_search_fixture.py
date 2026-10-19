from fastapi.testclient import TestClient
import pytest

from osworld_agent.viz.server import app


def test_rehearsal_requires_title_submission_and_labels_its_provenance():
    client = TestClient(app)
    empty = client.get('/demo/paper-search')
    assert empty.status_code == 200
    assert 'LOCAL REPLAY' in empty.text and 'NOT LIVE arxiv.org SEARCH' in empty.text
    assert 'https://export.arxiv.org/abs/1706.03762' not in empty.text

    result = client.get('/demo/paper-search', params={'query': 'Attention Is All You Need'})
    assert result.status_code == 200
    assert result.text.index('A Survey of Attention Mechanisms') < result.text.index('>Attention Is All You Need</a>')
    assert 'https://export.arxiv.org/abs/1706.03762' in result.text
    assert 'Ashish Vaswani' in result.text
    assert 'LOCAL REPLAY' in result.text


def test_search_query_is_escaped():
    result = TestClient(app).get('/demo/paper-search', params={'query': '<script>alert(1)</script>'})
    assert '<script>' not in result.text
    assert '&lt;script&gt;' in result.text


@pytest.mark.parametrize(('title', 'paper_id'), [
    ('BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', '1810.04805'),
    ('Deep Residual Learning for Image Recognition', '1512.03385'),
])
def test_other_rehearsal_titles_lead_to_their_distinct_official_papers(title, paper_id):
    client = TestClient(app)
    assert paper_id not in client.get('/demo/paper-search').text
    result = client.get('/demo/paper-search', params={'query': title})
    assert result.status_code == 200
    assert result.text.index('Related demonstration result') < result.text.index('>Paper record</span>')
    assert f'https://export.arxiv.org/abs/{paper_id}' in result.text
    assert title in result.text and 'LOCAL REPLAY' in result.text
