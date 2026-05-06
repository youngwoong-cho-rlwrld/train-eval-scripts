#!/bin/bash
# Interactive job & log viewer.
# Log views auto-tail; press Ctrl-C to return to the menu.
#
# Initial defaults can be overridden by env vars or via the in-menu "d" option.
#   SQF_LOG_DIR     where slurm .out/.err logs live   (default: $HOME/logs)
#   SQF_EXP_DIR     where per-experiment logs live    (default: $HOME/scripts)
#                   (server logs at $EXP_DIR/<exp>/logs/server_*.log)

LOG_DIR="${SQF_LOG_DIR:-$HOME/logs}"
EXP_DIR="${SQF_EXP_DIR:-$HOME/scripts}"

show_status() {
    clear
    echo '=== sacct (last 24h) ==='
    sacct -u $(whoami) -X -S $(date -d '24 hours ago' +%Y-%m-%dT%H:%M) \
          --format=JobID,JobName%30,State,ExitCode,Start,Elapsed,NodeList
    echo
    echo '=== current squeue ==='
    squeue -u $(whoami) -o '%.10i %.20j %.10P %.8T %.10M %.10L %R'
    echo
    read -n 1 -s -p 'press any key to return to menu...'
}

show_gpus() {
    clear
    local data
    data=$(sinfo -N -h --Format='NodeList:18,Partition:25,StateCompact:8,Gres:25,GresUsed:30' 2>/dev/null)

    echo '=== Free GPUs by partition (where you can submit right now) ==='
    printf '  %-25s %s\n' 'PARTITION' 'FREE_GPUs'
    echo "$data" | awk '
        function gc(s,    n) { n=0; if (match(s, /:[0-9]+\(/)) n = substr(s, RSTART+1, RLENGTH-2) + 0; return n }
        $3 !~ /-$/ && $3 !~ /^(down|drain|drng|fail|maint|resv|unk|comp|boot|plnd|planned)/ {
            free = gc($4) - gc($5)
            if (free < 0) free = 0
            sums[$2] += free
            if (!($2 in sums)) sums[$2] = 0
        }
        END {
            for (p in sums) if (sums[p] > 0) printf "  %-25s %d\n", p, sums[p]
        }
    ' | sort -k2,2nr

    echo
    echo '=== Per-node free GPUs (skipped: draining/down) ==='
    printf '  %-18s %-25s %-8s %s\n' 'NODE' 'PARTITION' 'STATE' 'FREE/TOTAL'
    echo "$data" | awk '
        function gc(s,    n) { n=0; if (match(s, /:[0-9]+\(/)) n = substr(s, RSTART+1, RLENGTH-2) + 0; return n }
        $3 !~ /-$/ && $3 !~ /^(down|drain|drng|fail|maint|resv|unk|comp|boot|plnd|planned)/ {
            total = gc($4); used = gc($5); free = total - used
            if (free > 0) printf "  %-18s %-25s %-8s %d/%d\n", $1, $2, $3, free, total
        }
    ' | sort -u
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
    echo
    read -n 1 -s -p 'press any key to return to menu...'
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
    echo '  d) Set directories'
    echo '  q) Quit'
    echo
    read -n 1 -s -p '> ' choice
    echo
    case "$choice" in
        1) show_status ;;
        2) show_log "$LOG_DIR/*.out" 'STDOUT' ;;
        3) show_log "$LOG_DIR/*.err" 'STDERR' ;;
        4) show_log "$EXP_DIR/*/logs/server_*.log" 'server' ;;
        5) show_gpus ;;
        d|D) set_dirs ;;
        q|Q) clear; exit 0 ;;
    esac
done
