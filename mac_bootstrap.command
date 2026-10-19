#!/bin/bash
# One-command setup on the user's Mac, followed by a loopback-only Studio tunnel.
set -euo pipefail
umask 077
[[ "$(uname -s)" == Darwin ]] || { echo '请在 Mac 终端执行这条命令。' >&2; exit 1; }
server=chenyuning@114.212.10.41
source_root=/data1/chenyuning/osworld_agent_framework_and_2trajectories/framework
# Use an app-managed copy, leaving the user's existing projects untouched.
state=${YAERO_MAC_STATE:-"$HOME/.yaero"}
root="$state/mac-agent"
mkdir -p "$state" "$root"
chmod 700 "$state"
connection_dir=$(mktemp -d /tmp/yaero-connect.XXXXXX)
control="$connection_dir/ssh"
cleanup() {
  ssh -S "$control" -O exit "$server" >/dev/null 2>&1 || true
  rm -rf "$connection_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'echo "准备未完成，请把最后的报错发给助手；不用重新手动配置。" >&2' ERR

echo '[1/4] 连接服务器并同步 Mac 执行端（保留已有个人项目）'
ssh -M -S "$control" -o ControlPersist=600 -fNT "$server"
rsync -az -e "ssh -S $control" \
  --exclude='.venv*' --exclude='venv' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='.git' --exclude='.env' \
  --exclude='.runtime' --exclude='local_runs' --exclude='runs' \
  --exclude='workspace' --exclude='downloads' \
  "$server:$source_root/" "$root/"

echo '[2/4] 自动准备 Python 和依赖（首次需要下载，之后复用）'
uv="$state/bin/uv"
if [[ ! -x "$uv" ]]; then
  mkdir -p "$state/bin"
  curl -fLsS --retry 2 --connect-timeout 15 --max-time 120 \
    https://astral.sh/uv/install.sh -o "$connection_dir/install-uv.sh"
  # Official installer options: https://docs.astral.sh/uv/reference/installer/
  UV_UNMANAGED_INSTALL="$state/bin" sh "$connection_dir/install-uv.sh"
fi
export UV_PYTHON_INSTALL_DIR="$state/python"
if [[ ! -x "$root/.venv-mac/bin/python" ]]; then
  "$uv" venv --python 3.12 "$root/.venv-mac"
fi
"$uv" pip install --python "$root/.venv-mac/bin/python" -r "$root/requirements-mac.txt"

echo '[3/4] 复用模型配置'
if [[ ! -e "$root/.env" ]]; then
  # Copy the user's existing credentials over SSH without displaying them.
  # Never overwrite an existing Mac configuration.
  if ssh -S "$control" "$server" "test -f '$source_root/.env'"; then
    scp -q -o "ControlPath=$control" "$server:$source_root/.env" "$root/.env"
    chmod 600 "$root/.env"
    echo '现有模型配置已通过 SSH 保存到 Mac 私有目录。'
  fi
fi

echo '[4/4] 启动 Mac 框架并建立连接'
echo "本地执行端：$root"
echo '保持当前终端打开，按 Ctrl+C 可断开连接。'
MAC_SSH_CONTROL_PATH="$control" MAC_PYTHON="$root/.venv-mac/bin/python" \
  bash "$root/mac_connect.command" "$server"
