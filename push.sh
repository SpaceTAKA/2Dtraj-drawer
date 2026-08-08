#!/bin/bash
# 簡易デプロイスクリプト: このリポジトリの変更をcommitしてGitHubにpushする。
# 使い方: このリポジトリのルートで `./push.sh` を実行するだけ。
# (実行権限が無い場合は先に `chmod +x push.sh` を一度だけ実行してください)
set -e
cd "$(dirname "$0")"
git add -A
if git diff --cached --quiet; then
  echo "コミットする新しい変更はありません（pushだけ試みます）。"
else
  git commit -m "update $(date '+%Y-%m-%d %H:%M:%S')"
fi
if ! git push; then
  echo "pushが拒否されました(リモートに未取得の変更があるようです)。取り込んで再試行します..."
  git pull --rebase origin main
  git push
fi
echo "push完了。GitHub Pagesの反映まで1分ほど待ってからiPadでリロードしてください。"
