#!/usr/bin/env bash
# Render the slides and build presentation.html.
#
# If the Windows render machine (RENDER_HOST, default user@192.168.50.210) answers over ssh with
# key login, the scenes are rendered there in parallel and the result is copied back into
# slides/; otherwise they are rendered here. presentation.html is always built here.
#
#   ./render.sh                     all scenes of scenes.txt, final quality (manim.cfg: 1080p30)
#   ./render.sh Results Conclusions only these scenes (presentation.html still has all of them)
#   ./render.sh -q l                quality l|m|h|p|k (480p15 ... 4K)
#   ./render.sh --local | --remote  force where to render (--remote fails if the host is down)
#   ./render.sh --no-html           skip building presentation.html
#
# The render machine has no LaTeX: new formulas are compiled here first (a quick pass that only
# builds the scenes, -s -ql), and media/Tex is sent along. One-time setup of the machine: README.md.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${RENDER_HOST:-user@192.168.50.210}"
RDIR='C:\Users\user\thesis_render'
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST")
VENV=.venv/bin
MODE=auto QUALITY="" HTML=1 SCENES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --local) MODE=local ;;
    --remote) MODE=remote ;;
    --no-html) HTML=0 ;;
    -q) QUALITY="$2"; shift ;;
    -q?) QUALITY="${1#-q}" ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) SCENES+=("$1") ;;
  esac
  shift
done
read -r -a ALL < scenes.txt
[ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
QFLAG=(); [ -n "$QUALITY" ] && QFLAG=("-q$QUALITY")

if [ "$MODE" != local ] && "${SSH[@]}" "if exist $RDIR\\venv\\Scripts\\manim-slides.exe (exit 0) else (exit 3)" 2>/dev/null; then
  WHERE=remote
elif [ "$MODE" = remote ]; then
  echo "render host $HOST unreachable or not set up (see README.md)" >&2; exit 1
else
  WHERE=local
fi
echo "==> rendering ${#SCENES[@]} scene(s) $WHERE"

if [ "$WHERE" = local ]; then
  "$VENV/manim-slides" render "${QFLAG[@]}" slides.py "${SCENES[@]}"
else
  # 1. compile new LaTeX here: build each scene with animations skipped, in a scratch copy whose
  #    media/Tex is this media/Tex (so slides/ is not touched). manim-slides then fails with
  #    "Cannot merge an empty list of files" (nothing was rendered): that error is expected.
  echo "==> compiling LaTeX locally"
  mkdir -p media/Tex
  T=$(mktemp -d)
  trap 'rm -rf "${T:?}"' EXIT
  cp slides.py helpers.py i18n_pt.py SPEAKER_NOTES.md manim.cfg "$T"/
  ln -s "$PWD/assets" "$T/assets"
  mkdir -p "$T/media"
  ln -s "$PWD/media/Tex" "$T/media/Tex"
  for s in "${SCENES[@]}"; do
    if ! out=$(cd "$T" && "$OLDPWD/$VENV/manim" -s -ql --progress_bar none slides.py "$s" 2>&1); then
      if ! grep -q "Cannot merge an empty list of files" <<<"$out"; then
        echo "$out" | tail -30; echo "scene $s failed while building" >&2; exit 1
      fi
    fi
  done

  # 2. send sources, assets and the LaTeX cache: only what changed since the last send to this
  #    host (delete the stamp in media/ to send everything again)
  stamp="media/.render_sent_${HOST//[^A-Za-z0-9]/_}"
  newstamp="$T/stamp"; touch "$newstamp"
  newer=(); if [ -f "$stamp" ]; then newer=(-newer "$stamp"); fi
  mapfile -t send < <(find slides.py helpers.py i18n_pt.py SPEAKER_NOTES.md manim.cfg scenes.txt \
      requirements.txt tools/render_remote.ps1 assets media/Tex -type f "${newer[@]}")
  echo "==> sending ${#send[@]} changed file(s) to $HOST"
  if [ ${#send[@]} -gt 0 ]; then
    printf '%s\n' "${send[@]}" | tar czf - -T - | "${SSH[@]}" "cd /d $RDIR && tar xzf -"
  fi
  cp -p "$newstamp" "$stamp"

  # 3. render there
  list=$(IFS=,; echo "${SCENES[*]}")
  "${SSH[@]}" "powershell -NoProfile -ExecutionPolicy Bypass -Command \"& $RDIR\\tools\\render_remote.ps1 -Scenes $list -Quality '$QUALITY'; exit \$LASTEXITCODE\""

  # 4. copy back slides/<scene>.json and slides/files/<scene>/, with / instead of \ in the paths
  echo "==> copying the rendered slides back"
  files=()
  for s in "${SCENES[@]}"; do files+=("slides/$s.json" "slides/files/$s"); rm -rf "slides/files/${s:?}"; done
  mkdir -p slides
  "${SSH[@]}" "cd /d $RDIR && tar cf - ${files[*]}" | tar xf - -m
  for s in "${SCENES[@]}"; do
    sed -i -E '/"(rev_)?file":/ s#\\\\#/#g' "slides/$s.json"
    # the remote folder keeps the videos of every past render: keep only the ones in use
    for f in "slides/files/$s"/*; do
      grep -qF "\"$f\"" "slides/$s.json" || rm -f "$f"
    done
  done
fi

if [ "$HTML" = 1 ]; then
  missing=(); for s in "${ALL[@]}"; do [ -f "slides/$s.json" ] || missing+=("$s"); done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "not building presentation.html: never rendered: ${missing[*]}" >&2; exit 1
  fi
  echo "==> building presentation.html"
  "$VENV/manim-slides" convert "${ALL[@]}" presentation.html --one-file --offline
fi
echo "==> done"
