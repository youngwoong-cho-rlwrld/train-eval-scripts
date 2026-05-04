#!/bin/bash
# Interactive job & log viewer.
# Log views auto-tail; press Ctrl-C to return to the menu.
#
# Initial defaults can be overridden by env vars or via the in-menu "d" option.
#   SQF_LOG_DIR     where slurm .out/.err logs live   (default: $HOME/logs)
#   SQF_EVAL_DIR    where eval results & server.log   (default: $HOME/eval_results)

LOG_DIR="${SQF_LOG_DIR:-$HOME/logs}"
EVAL_DIR="${SQF_EVAL_DIR:-$HOME/eval_results}"

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

set_dirs() {
    clear
    echo '=== Set directories ==='
    echo '(leave empty + Enter to keep current; ~ is expanded)'
    echo
    local new_log new_eval
    read -e -p "log dir  [$LOG_DIR]: " new_log
    if [ -n "$new_log" ]; then
        LOG_DIR="${new_log/#\~/$HOME}"
    fi
    read -e -p "eval dir [$EVAL_DIR]: " new_eval
    if [ -n "$new_eval" ]; then
        EVAL_DIR="${new_eval/#\~/$HOME}"
    fi
    echo
    echo 'Now using:'
    echo "  log dir:  $LOG_DIR"
    echo "  eval dir: $EVAL_DIR"
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
    echo "  eval dir: $EVAL_DIR"
    echo
    echo '  1) Job status (sacct + squeue)'
    echo '  2) STDOUT (.out)'
    echo '  3) STDERR (.err)'
    echo '  4) Isaac Sim server.log'
    echo '  d) Set directories'
    echo '  q) Quit'
    echo
    read -n 1 -s -p '> ' choice
    echo
    case "$choice" in
        1) show_status ;;
        2) show_log "$LOG_DIR/*.out" 'STDOUT' ;;
        3) show_log "$LOG_DIR/*.err" 'STDERR' ;;
        4) show_log "$EVAL_DIR/*/*/server.log" 'server.log' ;;
        d|D) set_dirs ;;
        q|Q) clear; exit 0 ;;
    esac
done
