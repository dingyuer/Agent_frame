"""Official identity adapter tests; no network or model requests."""
import pytest

from osworld_agent.paper_resolver import (PaperBinding, arxiv_id_from_url, parse_arxiv_metadata,
    query_matches_title, resolve_arxiv_candidates)


TITLE = 'RoboClaw: An Example Framework for Robotics'
IDENTIFIER = '2603.11558'


def metadata_html(title=TITLE, identifier=IDENTIFIER):
    return (f'<meta name="citation_title" content="{title}">'
            '<meta name="citation_author" content="Example Author">'
            f'<meta name="citation_arxiv_id" content="{identifier}">'
            f'<meta name="citation_pdf_url" content="https://arxiv.org/pdf/{identifier}">')


def fixture_selection(query, candidates, **kwargs):
    """Predetermined semantic fixture, not a production title-based fallback."""
    matches = [item for item in candidates if query_matches_title(query, item.full_title)]
    if len(matches) != 1:
        return {'status': 'uncertain', 'selected_id': None, 'reason': 'fixture has no unique selected target', 'evidence': []}
    item = matches[0]
    return {'status': 'matched', 'selected_id': item.canonical_id, 'reason': 'fixture selects this observed target',
            'evidence': [{'candidate_id': item.canonical_id, 'field': 'full_title', 'quote': item.full_title}]}


def resolve(query='Roboclaw', identifier=IDENTIFIER, title=TITLE, extra_metadata=''):
    return resolve_arxiv_candidates(query, ['https://arxiv.org/abs/' + identifier],
        fetcher=lambda url, timeout: {'url': url, 'text': metadata_html(title, identifier) + extra_metadata},
        identity_selector=fixture_selection)


def test_official_metadata_binds_short_alias_without_substring_matching():
    result = resolve()
    assert result.status == 'resolved'
    assert result.binding.full_title == TITLE
    assert result.binding.pdf_url == 'https://arxiv.org/pdf/' + IDENTIFIER
    assert len(result.binding.evidence_hash) == 64
    assert '_proof' not in result.binding.to_dict()
    assert not query_matches_title('Claw', TITLE)
    assert not query_matches_title('Roboclaw', 'A Comparison of RoboClaw Methods')
    assert resolve('Claw').status == 'unavailable'


@pytest.mark.parametrize('url', [
    'https://arxiv.org.attacker.test/abs/2603.11558', 'https://user@arxiv.org/abs/2603.11558',
    'file:///abs/2603.11558', 'https://arxiv.org/search/?query=2603.11558',
    'https://arxiv.org:1234/abs/2603.11558', 'https://example.test/abs/2603.11558'])
def test_only_observed_official_paper_url_can_start_resolution(url):
    assert arxiv_id_from_url(url) is None
    result = resolve_arxiv_candidates('Roboclaw', [url], fetcher=lambda *_args, **_kw: pytest.fail('unexpected fetch'))
    assert result.status == 'unavailable'


def test_dict_or_arbitrary_page_text_is_not_trusted_metadata():
    result = resolve_arxiv_candidates('Roboclaw', ['https://arxiv.org/abs/' + IDENTIFIER],
        metadata={'source_url': 'https://arxiv.org/abs/' + IDENTIFIER, 'title': TITLE},
        fetcher=lambda url, timeout: {'url': url, 'text': '<h1>' + TITLE + '</h1>'})
    assert result.status == 'unavailable'
    with pytest.raises(ValueError):
        parse_arxiv_metadata('https://example.test/abs/' + IDENTIFIER, metadata_html())


def test_conflicting_or_redirected_metadata_is_rejected():
    for body in (metadata_html() + '<meta name="citation_title" content="Different">',
                 metadata_html().replace('/pdf/' + IDENTIFIER, '/pdf/2603.99999')):
        with pytest.raises(ValueError):
            parse_arxiv_metadata('https://arxiv.org/abs/' + IDENTIFIER, body)
    result = resolve_arxiv_candidates('Roboclaw', ['https://arxiv.org/abs/' + IDENTIFIER],
        fetcher=lambda url, timeout: {'url': 'https://evil.test/abs/' + IDENTIFIER, 'text': metadata_html()})
    assert result.status == 'unavailable'


def test_multiple_matching_candidates_are_ambiguous_and_failures_do_not_establish_uniqueness():
    urls = ['https://arxiv.org/abs/' + identifier for identifier in (IDENTIFIER, '2603.99999')]
    def fetch(url, timeout):
        return {'url': url, 'text': metadata_html(identifier=url.rsplit('/', 1)[-1])}
    assert resolve_arxiv_candidates('Roboclaw', urls, fetcher=fetch, identity_selector=fixture_selection).status == 'unavailable'
    def partial(url, timeout):
        if url.endswith('99999'):
            raise TimeoutError()
        return fetch(url, timeout)
    assert resolve_arxiv_candidates('Roboclaw', urls, fetcher=partial).status == 'unavailable'
    assert resolve_arxiv_candidates('Roboclaw', urls, fetcher=fetch, max_candidates=1).status == 'ambiguous'


def test_typed_metadata_cache_avoids_request_and_timeout_is_bounded():
    url = 'https://arxiv.org/abs/' + IDENTIFIER
    evidence = parse_arxiv_metadata(url, metadata_html())
    result = resolve_arxiv_candidates('Roboclaw', [url], metadata=evidence,
        fetcher=lambda *_args, **_kw: pytest.fail('cached evidence should avoid fetch'), identity_selector=fixture_selection)
    assert result.status == 'resolved'
    timeouts = []
    def fetch(source, timeout):
        timeouts.append(timeout)
        return {'url': source, 'text': metadata_html()}
    resolve_arxiv_candidates('Roboclaw', [url], fetcher=fetch, timeout=999, total_timeout=1)
    assert 0 < timeouts[0] <= 1


def test_metadata_qualifiers_are_normalized_and_carried_by_binding():
    binding = resolve(extra_metadata='<meta name="citation_doi" content=" https://doi.org/10.1234/TEST ">'
                      '<meta name="citation_date" content="2026/03/12">').binding
    assert binding.dois == ('10.1234/test',)
    assert binding.publication_years == (2026,)
    assert binding.to_dict()['publication_years'] == (2026,)


def test_response_after_total_deadline_is_not_accepted(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('osworld_agent.paper_resolver.time.monotonic', lambda: clock[0])
    def late(url, timeout):
        clock[0] = 2.0
        return {'url': url, 'text': metadata_html()}
    result = resolve_arxiv_candidates('Roboclaw', ['https://arxiv.org/abs/' + IDENTIFIER],
                                     fetcher=late, total_timeout=1)
    assert result.status == 'unavailable'
    assert result.reason_code == 'resolution_timeout'


def test_resolution_reason_codes_distinguish_uncertain_evidence_from_mismatch():
    assert resolve_arxiv_candidates('', []).reason_code == 'invalid_query'
    assert resolve_arxiv_candidates('RoboClaw', []).reason_code == 'no_observed_candidate'
    many = ['https://arxiv.org/abs/2603.' + str(i).zfill(5) for i in range(8)]
    assert resolve_arxiv_candidates('RoboClaw', many).reason_code == 'too_many_candidates'
    assert resolve().reason_code == 'verified_identity'
    assert resolve('Different title').reason_code == 'semantic_uncertain'
    failed = resolve_arxiv_candidates('RoboClaw', ['https://arxiv.org/abs/' + IDENTIFIER],
        fetcher=lambda *args, **kwargs: (_ for _ in ()).throw(OSError('offline')))
    assert failed.status == 'unavailable' and failed.reason_code == 'metadata_fetch_failed'
    invalid = resolve_arxiv_candidates('RoboClaw', ['https://arxiv.org/abs/' + IDENTIFIER],
        fetcher=lambda url, **kwargs: {'url': url, 'text': '<html>No citation metadata</html>'})
    assert invalid.status == 'unavailable' and invalid.reason_code == 'metadata_invalid'
