#!/bin/bash
# 簡易デプロイスクリプト: このリポジトリの変更をcommitしてGitHubにpushする。
# 使い方: このリポジトリのルートで `./push.sh` を実行するだけ。
# (実行権限が無い場合は先に `chmod +x push.sh` を一度だけ実行してください)
set -e
cd "$(dirname "$0")"
git add -A
if git diff --cached --quiet; then
  echo "変更はありません。"
  exit 0
fi
git commit -m "update $(date '+%Y-%m-%d %H:%M:%S')"
git push
echo "push完了。GitHub Pagesの反映まで1分ほど待ってからiPadでリロードしてください。"
