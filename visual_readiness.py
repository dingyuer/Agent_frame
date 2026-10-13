"""Conservative PDF render heuristics; readiness is never task completion."""
from urllib.parse import urlsplit
from .browser_feedback import observation_url


def is_pdf_observation(obs):
    if obs is None or (obs.info or {}).get('active_surface') == 'desktop':
        return False
    info = obs.info or {}
    blocks = [b for b in obs.context if b.get('kind') == 'browser_ax']
    try:
        path = urlsplit(observation_url(obs)).path.lower()
    except ValueError:
        path = ''
    return any(b.get('document_kind') in {'pdf', 'pdf_viewer'} or 'application/pdf' in str(b.get('content_type', ''))
               for b in [info, *blocks]) or '/pdf/' in path or path.endswith('.pdf')


def content_thumbnail(screenshot):
    """Ignore browser/PDF toolbars; uniform black, white and gray are not ready."""
    try:
        from PIL import Image, ImageStat
        from io import BytesIO
        import base64
        if isinstance(screenshot, str) and screenshot.startswith('data:'):
            screenshot = Image.open(BytesIO(base64.b64decode(screenshot.split(',', 1)[1])))
        elif isinstance(screenshot, bytes):
            screenshot = Image.open(BytesIO(screenshot))
        elif isinstance(screenshot, str):
            screenshot = Image.open(screenshot)
        width, height = screenshot.size
        if width < 20 or height < 20:
            return None
        crop = screenshot.convert('L').crop((int(width * .15), int(height * .2), int(width * .85), int(height * .85))).resize((96, 96))
        return crop if ImageStat.Stat(crop).stddev[0] >= 3 else None
    except Exception:
        return None


def stable_content(previous, current):
    if previous is None or current is None:
        return False
    from PIL import ImageChops, ImageStat
    return ImageStat.Stat(ImageChops.difference(previous, current)).mean[0] <= 1.5
