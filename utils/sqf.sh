#!/bin/bash
# Interactive job & log viewer.
# Log views auto-tail; press Ctrl-C to return to the menu.
#
# Defaults can be overridden by (in order of precedence):
#   1. env vars SQF_LOG_DIR / SQF_EXP_DIR
#   2. saved config at SQF_CONFIG (default $HOME/.config/sqf/config)
#   3. the in-menu "d" option (saved on exit of that menu)
# On first run, sqf prompts and persists if either dir does not exist.
#
# Variables:
#   SQF_LOG_DIR     where slurm .out/.err logs live   (default: $HOME/logs)
#   SQF_EXP_DIR     where per-experiment logs live    (default: $HOME/train-eval-scripts/experiments)
#                   (server logs at $EXP_DIR/<exp>/logs/server_*.log)

# Ensure Slurm CLI is reachable. On skt the binaries live at /opt/slurm/bin
# which is only on PATH for login shells (via /etc/profile.d/aws-pcluster-env.sh);
# scripts and non-interactive shells need this prepended explicitly.
[ -d /opt/slurm/bin ] && export PATH="/opt/slurm/bin:$PATH"

LOG_DIR="${SQF_LOG_DIR:-$HOME/logs}"

# Auto-detect train-eval-scripts/experiments/ on the current cluster, fall back to ~/scripts.
if [ -d "$HOME/train-eval-scripts/experiments" ]; then
    _DEFAULT_EXP_DIR="$HOME/train-eval-scripts/experiments"
else
    _DEFAULT_EXP_DIR="$HOME/scripts"
fi
EXP_DIR="${SQF_EXP_DIR:-$_DEFAULT_EXP_DIR}"

# ── Persist user-chosen paths across sessions ──
# Order of precedence (first wins):
#   1. SQF_LOG_DIR / SQF_EXP_DIR env vars (handled above)
#   2. ~/.config/sqf/config — written by the in-menu "d" option or first-run prompt
#   3. auto-detected defaults
SQF_CONFIG="${SQF_CONFIG:-$HOME/.config/sqf/config}"
if [ -f "$SQF_CONFIG" ]; then
    # source uses the persisted SQF_LOG_DIR / SQF_EXP_DIR only when env didn't already set them
    while IFS= read -r line; do
        case "$line" in
            SQF_LOG_DIR=*) [ -z "${SQF_LOG_DIR:-}" ] && eval "$line"; LOG_DIR="${SQF_LOG_DIR:-$LOG_DIR}" ;;
            SQF_EXP_DIR=*) [ -z "${SQF_EXP_DIR:-}" ] && eval "$line"; EXP_DIR="${SQF_EXP_DIR:-$EXP_DIR}" ;;
        esac
    done < "$SQF_CONFIG"
fi

_save_sqf_config() {
    mkdir -p "$(dirname "$SQF_CONFIG")"
    cat > "$SQF_CONFIG" <<EOF
SQF_LOG_DIR="$LOG_DIR"
SQF_EXP_DIR="$EXP_DIR"
EOF
}

# ── First-run prompt: if either path is not a directory, ask the user once. ──
if [ ! -d "$LOG_DIR" ] || [ ! -d "$EXP_DIR" ]; then
    echo "=== sqf first-time setup ==="
    echo "(saved to $SQF_CONFIG; can be re-set anytime via the \"d\" menu option)"
    echo
    if [ ! -d "$LOG_DIR" ]; then
        read -e -p "Slurm log dir [$LOG_DIR]: " _ans
        [ -n "$_ans" ] && LOG_DIR="${_ans/#\~/$HOME}"
    fi
    if [ ! -d "$EXP_DIR" ]; then
        read -e -p "Experiment dir [$EXP_DIR]: " _ans
        [ -n "$_ans" ] && EXP_DIR="${_ans/#\~/$HOME}"
    fi
    _save_sqf_config
    echo "Saved. (Edit $SQF_CONFIG or use the \"d\" menu option to change later.)"
    echo
    read -n 1 -s -p "press any key to continue..."
fi

show_status() {
    clear
    echo '=== sacct (last 24h) ==='
    sacct -u "$(whoami)" -X -P -S "$(date -d '24 hours ago' +%Y-%m-%dT%H:%M)" \
          --format=JobID,JobName,State,ExitCode,Start,Elapsed,NodeList \
        | column -t -s '|'
    echo
    echo '=== current squeue ==='
    squeue -u "$(whoami)" -o '%i|%j|%P|%T|%M|%L|%R' | column -t -s '|'
    echo
    read -n 1 -s -p 'press any key to return to menu...'
}

show_gpus() {
    clear
    # Wide column widths so even long node/partition names (skt: l40s-gpu-st-
    # g6e-12xl-debug-3, rlwrld-gpu_background) get >=2 spaces of padding,
    # which lets awk -F '  +' split each line cleanly. Column widths in the
    # final printed tables are computed dynamically by `column -t`.
    local fmt='NodeList:40,Partition:30,StateCompact:10,Gres:25,GresUsed:30'
    local data
    data=$(sinfo -N -h --Format="$fmt" 2>/dev/null)

    echo '=== Free GPUs by partition (where you can submit right now) ==='
    {
        printf 'PARTITION\tFREE_GPUs\n'
        echo "$data" | awk -F '  +' '
            function gc(s,    n) {
                # Extract GPU count from a Gres / GresUsed string.
                # Handles both kakao-style "gpu:8(S:0-1)" / "gpu:(null):8(IDX:0-7)"
                # and skt-style "gpu:l40s:4" (no trailing parens).
                n = 0
                if (match(s, /[0-9]+\(/))      n = substr(s, RSTART, RLENGTH-1) + 0
                else if (match(s, /:[0-9]+$/)) n = substr(s, RSTART+1, RLENGTH-1) + 0
                return n
            }
            { sub(/^ +/, "") }
            # Accept idle/mix and their cloud "~" variants (idle~ = powered down,
            # auto-spins-up on allocation — the user can still submit there).
            $3 ~ /^(idle|mix)~?$/ {
                free = gc($4) - gc($5); if (free < 0) free = 0
                sums[$2] += free
                if (!($2 in sums)) sums[$2] = 0
            }
            END {
                for (p in sums) if (sums[p] > 0) printf "%s\t%d\n", p, sums[p]
            }
        ' | sort -t$'\t' -k2,2nr
    } | column -t -s $'\t'

    echo
    echo '=== Per-node free GPUs (idle/mix; ~ = cloud node, will spin up on submit) ==='
    {
        printf 'NODE\tPARTITION\tSTATE\tFREE/TOTAL\n'
        echo "$data" | awk -F '  +' '
            function gc(s,    n) {
                # Extract GPU count from a Gres / GresUsed string.
                # Handles both kakao-style "gpu:8(S:0-1)" / "gpu:(null):8(IDX:0-7)"
                # and skt-style "gpu:l40s:4" (no trailing parens).
                n = 0
                if (match(s, /[0-9]+\(/))      n = substr(s, RSTART, RLENGTH-1) + 0
                else if (match(s, /:[0-9]+$/)) n = substr(s, RSTART+1, RLENGTH-1) + 0
                return n
            }
            { sub(/^ +/, "") }
            # Accept idle/mix and their cloud "~" variants (idle~ = powered down,
            # auto-spins-up on allocation — the user can still submit there).
            $3 ~ /^(idle|mix)~?$/ {
                total = gc($4); used = gc($5); free = total - used
                if (free > 0) printf "%s\t%s\t%s\t%d/%d\n", $1, $2, $3, free, total
            }
        ' | sort -u
    } | column -t -s $'\t'
    echo
    read -n 1 -s -p 'press any key to return to menu...'
}

set_dirs() {
    clear
    echo '=== Set directories ==='
    echo '(leave empty + Enter to keep current; ~ is expanded)'
    echo
    local new_log new_exp
    read -e -p "log dir  [$LOG_DIR]: " new_log
    if [ -n "$new_log" ]; then
        LOG_DIR="${new_log/#\~/$HOME}"
    fi
    read -e -p "exp dir  [$EXP_DIR]: " new_exp
    if [ -n "$new_exp" ]; then
        EXP_DIR="${new_exp/#\~/$HOME}"
    fi
    echo
    echo 'Now using:'
    echo "  log dir:  $LOG_DIR"
    echo "  exp dir:  $EXP_DIR"
    _save_sqf_config
    echo "(saved to $SQF_CONFIG)"
    echo
    read -n 1 -s -p 'press any key to return to menu...'
}

cancel_jobs() {
    clear
    local rows=()
    while IFS= read -r line; do
        rows+=("$line")
    done < <(squeue -h -u "$(whoami)" -o '%i|%j|%T|%M|%R' 2>/dev/null)

    if [ ${#rows[@]} -eq 0 ]; then
        echo '(no jobs to cancel)'
        echo
        read -n 1 -s -p 'press any key to return to menu...'
        return
    fi

    {
        printf 'IDX\tJOBID\tNAME\tSTATE\tELAPSED\tREASON/NODE\n'
        local i=0
        for line in "${rows[@]}"; do
            i=$((i+1))
            printf '%d\t%s\n' "$i" "$(echo "$line" | tr '|' '\t')"
        done
    } | column -t -s $'\t'
    echo
    echo "  Enter index/indexes (e.g. '1' or '1 3'), 'a' for ALL, 'q' to cancel"
    read -p '> ' choice

    if [[ "$choice" =~ ^[qQ]$ || -z "$choice" ]]; then
        return
    fi

    local to_cancel=()
    if [[ "$choice" =~ ^[aA]$ ]]; then
        for line in "${rows[@]}"; do
            to_cancel+=("$(echo "$line" | awk -F '|' '{print $1}')")
        done
    else
        for n in $choice; do
            if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le ${#rows[@]} ]; then
                to_cancel+=("$(echo "${rows[$((n-1))]}" | awk -F '|' '{print $1}')")
            fi
        done
    fi

    if [ ${#to_cancel[@]} -eq 0 ]; then
        echo 'no valid selection'
        sleep 1
        return
    fi

    echo
    echo "Will scancel: ${to_cancel[*]}"
    read -p 'confirm? [y/N] ' confirm
    if [[ "$confirm" =~ ^[yY]$ ]]; then
        scancel "${to_cancel[@]}" && echo 'cancelled' || echo 'scancel failed'
    else
        echo 'aborted'
    fi
    sleep 1.5
}

show_log() {
    local pattern="$1"
    local label="$2"

    local files=()
    while IFS= read -r f; do
        files+=("$f")
    done < <(ls -t $pattern 2>/dev/null)

    if [ ${#files[@]} -eq 0 ]; then
        clear
        echo "(no $label files yet under: $pattern)"
        read -n 1 -s -p 'press any key to return to menu...'
        return
    fi

    local picked
    if [ ${#files[@]} -eq 1 ]; then
        picked="${files[0]}"
    else
        clear
        echo "Pick $label log:"
        echo
        local i=0
        for f in "${files[@]}"; do
            i=$((i+1))
            local fname mtime jobid state
            fname=$(basename "$f")
            mtime=$(date -r "$f" '+%m-%d %H:%M' 2>/dev/null)
            jobid=$(echo "$fname" | sed -nE 's/.*_([0-9]+)\.[^.]+$/\1/p')
            state=''
            if [ -n "$jobid" ]; then
                state=$(sacct -j "$jobid" -X --noheader -P --format=State,NodeList 2>/dev/null \
                        | head -1 | tr '|' ' ')
            fi
            printf '  %2d) %-55s  %s  %s\n' "$i" "$fname" "$mtime" "$state"
        done
        echo '   q) cancel'
        echo
        read -p '> ' choice
        if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
            return
        fi
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#files[@]} ]; then
            picked="${files[$((choice-1))]}"
        else
            echo 'invalid choice'
            sleep 1
            return
        fi
    fi

    clear
    printf '=== %s: %s ===\n' "$label" "$picked"
    printf '(Ctrl-C to return to menu)\n\n'
    trap 'true' INT
    tail -n +1 -f "$picked"
    trap - INT
}

while true; do
    clear
    echo '================================'
    echo '  Job & log viewer'
    echo '================================'
    echo "  log dir:  $LOG_DIR"
    echo "  exp dir:  $EXP_DIR"
    echo
    echo '  1) Job status (sacct + squeue)'
    echo '  2) STDOUT (.out)'
    echo '  3) STDERR (.err)'
    echo '  4) Isaac Sim server log'
    echo '  5) GPU availability'
    echo '  c) Cancel job(s)'
    echo '  d) Set directories'
    echo '  q) Quit'
    echo
    echo '--------------------------------'
    echo '  Features are not complete and there may be bugs.'
    echo '  If anything found, please create an issue at:'
    echo '  https://github.com/youngwoong-cho-rlwrld/train-eval-scripts/issues'
    echo
    read -n 1 -s -p '> ' choice
    echo
    case "$choice" in
        1) show_status ;;
        2) show_log "$LOG_DIR/*.out" 'STDOUT' ;;
        3) show_log "$LOG_DIR/*.err" 'STDERR' ;;
        4) show_log "$EXP_DIR/*/logs/server_*.log" 'server' ;;
        5) show_gpus ;;
        c|C) cancel_jobs ;;
        d|D) set_dirs ;;
        q|Q) clear; exit 0 ;;
    esac
done
