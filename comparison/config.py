"""Local settings. Secrets are never returned through the settings endpoint."""
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent

def read_env(path):
    values = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values

def flag(value):
    return str(value).lower() in ('true', '1', 'yes')

def endpoint(url, suffix):
    url = url.strip().rstrip('/')
    if not url:
        return ''
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('地址必须是 http(s) URL，认证请填写密钥字段')
    if url.endswith(suffix):
        return url
    return url + suffix[3:] if url.endswith('/v1') else url + suffix

@dataclass(frozen=True)
class Settings:
    jev_url: str = ''
    jev_model: str = 'd3-flash'
    jev_api_key: str = field(default='', repr=False)
    vlm_url: str = ''
    vlm_model: str = 'qwen3.6-flash'
    vlm_api_key: str = field(default='', repr=False)
    vlm_enable_thinking: bool = False
    vlm_thinking_mode: str = 'dashscope'
    vlm_json_mode: bool = True
    vlm_trust_env: bool = True
    jev_trust_env: bool = False
    timeout: float = 90
    threshold: float = .80
    margin: float = .15

    @classmethod
    def load(cls):
        env = {**read_env(ROOT / '.env'), **os.environ}
        defaults = cls()
        data = {}
        aliases = {'timeout':'MODEL_TIMEOUT', 'threshold':'JEV_THRESHOLD', 'margin':'JEV_MARGIN'}
        for name in cls.__dataclass_fields__:
            value = env.get(aliases.get(name, name.upper()), getattr(defaults, name))
            if isinstance(getattr(defaults, name), bool):
                value = flag(value)
            elif name in aliases:
                value = float(value)
            data[name] = value
        return cls(**data).validated()

    def validated(self):
        if self.vlm_thinking_mode not in ('dashscope', 'chat_template', 'omit'):
            raise ValueError('思考参数模式必须是 dashscope / chat_template / omit')
        if not 1 <= self.timeout <= 300 or not .5 <= self.threshold <= 1 or not 0 <= self.margin <= 1:
            raise ValueError('超时范围 1–300 秒，阈值 0.5–1，差值 0–1')
        if not self.jev_model.strip() or not self.vlm_model.strip():
            raise ValueError('模型名不能为空')
        return replace(self, jev_url=endpoint(self.jev_url, '/v1/systemone'), vlm_url=endpoint(self.vlm_url, '/v1/chat/completions'))

    def updated(self, patch):
        if not isinstance(patch, dict) or set(patch) - set(self.__dataclass_fields__):
            raise ValueError('存在未知配置项')
        clean = {}
        for key, value in patch.items():
            if key.endswith('api_key') and value == '':
                continue  # Empty password inputs preserve the server-side secret.
            old = getattr(self, key)
            if isinstance(old, bool):
                if type(value) is not bool:
                    raise ValueError('布尔配置类型不正确')
            elif key in ('timeout', 'threshold', 'margin'):
                if type(value) not in (float, int):
                    raise ValueError('数值配置类型不正确')
            elif not isinstance(value, str) or len(value) > 2000:
                raise ValueError('文本配置类型不正确或过长')
            clean[key] = value
        return replace(self, **clean).validated()

    def public(self):
        return {**{k:v for k,v in vars(self).items() if not k.endswith('api_key')},
                'jev_key_configured': bool(self.jev_api_key), 'vlm_key_configured': bool(self.vlm_api_key)}
