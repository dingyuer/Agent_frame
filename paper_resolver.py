"""Bounded arXiv identity resolution from official metadata, never model prose.

The browser supplies observed candidate URLs. This adapter fetches their fixed
official abstract endpoints (or reuses typed metadata evidence from this adapter).
It cannot turn an arbitrary page title or model-produced dict into a binding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from html.parser import HTMLParser
import re
import time
import unicodedata
from urllib.parse import urlsplit

ADAPTER_VERSION = 'arxiv-citation-v1'
_ID = r'(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?'
_SEAL = object()


def arxiv_id_from_url(value):
    if not isinstance(value, str):
        return None
    try:
        url = urlsplit(value)
        if (url.scheme not in ('http', 'https') or url.hostname not in ('arxiv.org', 'www.arxiv.org')
                or url.username or url.password or url.port not in (None, 80, 443)):
            return None
        match = re.fullmatch(r'/(?:abs|pdf|html)/(' + _ID + r')(?:\.pdf)?/?', url.path, re.I)
        return match.group(1) if match else None
    except ValueError:
        return None


def _base_id(value):
    return re.sub(r'v\d+$', '', value, flags=re.I).lower()


def _text_key(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def query_matches_title(query, title):
    """Only a complete title or a complete colon-delimited leading alias."""
    if not isinstance(query, str) or not isinstance(title, str):
        return False
    query, title = _text_key(query), _text_key(title)
    return bool(query and (query == title or query == title.split(':', 1)[0].strip()))


@dataclass(frozen=True)
class ArxivMetadata:
    canonical_id: str
    full_title: str
    authors: tuple[str, ...]
    source_url: str
    pdf_url: str
    evidence_hash: str
    retrieved_at: float
    adapter_version: str = ADAPTER_VERSION
    dois: tuple[str, ...] = ()
    publication_years: tuple[int, ...] = ()
    _proof: object = None


@dataclass(frozen=True)
class PaperBinding:
    query: str
    canonical_id: str
    full_title: str
    authors: tuple[str, ...]
    source_url: str
    pdf_url: str
    evidence_hash: str
    retrieved_at: float
    adapter_version: str = ADAPTER_VERSION
    dois: tuple[str, ...] = ()
    publication_years: tuple[int, ...] = ()
    _proof: object = None

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if key != '_proof'}


@dataclass(frozen=True)
class PaperResolution:
    status: str
    binding: PaperBinding | None = None
    reason: str = ''
    candidates: tuple[str, ...] = ()


def is_verified_binding(binding):
    return isinstance(binding, PaperBinding) and binding._proof is _SEAL


class _CitationParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'meta':
            return
        attrs = dict(attrs)
        name = (attrs.get('name') or '').lower()
        if name.startswith('citation_') and isinstance(attrs.get('content'), str):
            self.fields.setdefault(name, []).append(attrs['content'])


def parse_arxiv_metadata(source_url, html, *, retrieved_at=None):
    """Parse bytes acquired from this official URL by a trusted transport.

    This function is an adapter boundary for browser response caches and tests;
    callers must pass the actual response body, not DOM/model-generated values.
    """
    identifier = arxiv_id_from_url(source_url)
    if not identifier or not urlsplit(source_url).path.startswith('/abs/'):
        raise ValueError('Official arXiv abstract response required')
    if not isinstance(html, str) or len(html.encode('utf-8')) > 2 * 1024 * 1024:
        raise ValueError('Metadata response exceeds bounded size')
    parser = _CitationParser()
    parser.feed(html)
    values = parser.fields
    titles = {v.strip() for v in values.get('citation_title', []) if v.strip()}
    pdfs = {v.strip() for v in values.get('citation_pdf_url', []) if v.strip()}
    authors = tuple(dict.fromkeys(v.strip() for v in values.get('citation_author', []) if v.strip()))
    if len(titles) != 1 or len(pdfs) != 1 or not authors:
        raise ValueError('Missing or ambiguous official citation metadata')
    title, pdf = next(iter(titles)), next(iter(pdfs))
    pdf_id = arxiv_id_from_url(pdf)
    if not pdf_id or _base_id(pdf_id) != _base_id(identifier) or '/pdf/' not in urlsplit(pdf).path:
        raise ValueError('PDF metadata does not identify the observed paper')
    abstract_ids = values.get('citation_arxiv_id', [])
    if any(_base_id(v.removeprefix('arXiv:').strip()) != _base_id(identifier) for v in abstract_ids):
        raise ValueError('Conflicting arXiv identifiers in citation metadata')
    if re.search(r'v\d+$', identifier, re.I) and pdf_id != identifier:
        raise ValueError('Requested arXiv version differs from PDF metadata')
    canonical = pdf_id if re.search(r'v\d+$', pdf_id, re.I) else identifier
    dois = tuple(sorted({re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', value.strip(), flags=re.I).lower()
        for value in values.get('citation_doi', []) if value.strip()}))
    if any(not re.fullmatch(r'10\.\d{4,9}/\S+', value) for value in dois):
        raise ValueError('Invalid official DOI metadata')
    dates = values.get('citation_date', []) + values.get('citation_publication_date', [])
    years = tuple(sorted({int(m.group(1)) for value in dates if (m := re.match(r'^(\d{4})(?:\D|$)', value.strip()))}))
    return ArxivMetadata(canonical, ' '.join(title.split()), authors,
        'https://arxiv.org/abs/' + identifier, 'https://arxiv.org/pdf/' + canonical,
        hashlib.sha256(html.encode('utf-8')).hexdigest(),
        time.time() if retrieved_at is None else float(retrieved_at),
        dois=dois, publication_years=years, _proof=_SEAL)


def _fetch_official(url, timeout):
    import requests
    deadline = time.monotonic() + timeout
    with requests.get(url, timeout=timeout, allow_redirects=False, stream=True) as response:
        if response.status_code != 200:
            raise ValueError('Official metadata endpoint did not return HTTP 200')
        body = bytearray()
        # read1 returns currently available bytes instead of waiting for a full
        # large chunk. Checking a deadline between reads prevents a trickling
        # response from resetting the read timeout indefinitely. The fallback
        # likewise yields each received byte on older urllib3 transports.
        read1 = getattr(response.raw, 'read1', None)
        chunks = iter(lambda: read1(65536, decode_content=True), b'') if read1 else response.iter_content(1)
        for chunk in chunks:
            if time.monotonic() > deadline:
                raise TimeoutError('Official metadata response exceeded its time budget')
            body.extend(chunk)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError('Official metadata response too large')
        return {'url': response.url, 'text': bytes(body).decode('utf-8'), 'status_code': 200}


def resolve_arxiv_candidates(query, observed_urls, *, fetcher=None, metadata=None,
                             max_candidates=4, timeout=5.0, total_timeout=12.0):
    """Resolve a unique matching candidate within a bounded observed set.

    Missing responses do not prove uniqueness. Candidate limits/partial failures
    yield unavailable rather than selecting whichever request happened to work.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        return PaperResolution('unavailable', reason='Invalid paper query')
    if isinstance(observed_urls, str):
        observed_urls = [observed_urls]
    identifiers = list(dict.fromkeys(i for u in observed_urls if (i := arxiv_id_from_url(u))))
    if not identifiers:
        return PaperResolution('unavailable', reason='No observed official paper URL')
    if len(identifiers) > max(1, min(int(max_candidates), 4)):
        return PaperResolution('ambiguous', reason='Too many observed candidates', candidates=tuple(identifiers))
    evidence = metadata if isinstance(metadata, (list, tuple)) else [metadata]
    cached = {m.canonical_id: m for m in evidence if isinstance(m, ArxivMetadata) and m._proof is _SEAL}
    matches, incomplete = {}, False
    deadline = time.monotonic() + max(0.01, min(float(total_timeout), 30.0))
    fetcher = fetcher or _fetch_official
    for identifier in identifiers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            incomplete = True
            break
        try:
            item = cached.get(identifier)
            if item is None:
                source = 'https://arxiv.org/abs/' + identifier
                response = fetcher(source, timeout=max(0.01, min(float(timeout), remaining, 5.0)))
                if isinstance(response, dict):
                    actual, html, status = response.get('url'), response.get('text'), response.get('status_code', 200)
                else:
                    actual, html, status = response.url, response.text, response.status_code
                if status != 200 or arxiv_id_from_url(actual) != identifier or not actual.startswith('https://arxiv.org/abs/'):
                    raise ValueError('Unexpected official metadata response identity')
                item = parse_arxiv_metadata(actual, html)
            if time.monotonic() > deadline:
                raise TimeoutError('Official identity resolution exceeded its time budget')
            if query_matches_title(query, item.full_title):
                matches[_base_id(item.canonical_id)] = item
        except Exception:
            incomplete = True
    if len(matches) > 1:
        return PaperResolution('ambiguous', reason='Multiple official papers match the query', candidates=tuple(matches))
    if incomplete or not matches:
        return PaperResolution('unavailable', reason='Official identity evidence is incomplete or does not match the query',
                               candidates=tuple(identifiers))
    item = next(iter(matches.values()))
    binding = PaperBinding(query.strip(), item.canonical_id, item.full_title, item.authors,
        item.source_url, item.pdf_url, item.evidence_hash, item.retrieved_at,
        dois=item.dois, publication_years=item.publication_years, _proof=_SEAL)
    return PaperResolution('resolved', binding=binding, candidates=(item.canonical_id,))
