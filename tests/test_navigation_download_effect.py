from osworld_agent.env import Observation
from osworld_agent.judge_evidence import effect_review


def test_navigation_download_is_verified_by_new_matching_browser_artifact():
    url = 'https://example.test/paper.pdf'
    before = Observation(info={'url': 'https://example.test/results', 'artifacts': []})
    artifact = {'artifact_id': 'new-transfer', 'url': url, 'completed': True}
    after = Observation(info={'url': before.info['url'], 'artifacts': [artifact]})
    execution = {'action': 'goto', 'args': {'url': url}, 'status': 'uncertain', 'dispatched': True}
    assert effect_review(before, after, execution)['status'] == 'passed'
    before.info['artifacts'] = [artifact]
    assert effect_review(before, after, execution)['status'] == 'unknown'


def test_unrelated_file_or_undispatched_navigation_cannot_prove_transfer():
    before = Observation(info={'url': 'https://example.test/results'})
    after = Observation(info={'url': before.info['url'], 'artifacts': [
        {'artifact_id': 'other', 'url': 'https://example.test/other.pdf', 'completed': True}]})
    execution = {'action': 'goto', 'args': {'url': 'https://example.test/paper.pdf'}, 'status': 'uncertain', 'dispatched': True}
    assert effect_review(before, after, execution)['status'] == 'unknown'
    execution['dispatched'] = False
    assert effect_review(before, after, execution)['status'] == 'failed'
