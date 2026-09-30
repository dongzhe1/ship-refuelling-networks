set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$ROOT/pipeline"
PY="${PYTHON:-python3}"
J="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
export PYTHONHASHSEED=0

run() { echo "--- $*"; (cd "$P" && "$PY" "$@"); }
abspath() { mkdir -p "$1" && (cd "$1" && pwd); }
has_pull() { [ -n "${1:-}" ] && ls "$1"/port_visits_*.csv.gz > /dev/null 2>&1; }

collect() {
    local out="$1" dest="$2"
    mkdir -p "$dest"
    find "$out/results" -maxdepth 1 -type f -exec cp {} "$dest/" \;
    local pair src
    for pair in methanol_pilot:pilot methanol_pilot_ext:pilot_ext identity_rerun:identity_rerun; do
        src="$out/${pair%%:*}/results"
        if [ -d "$src" ]; then mkdir -p "$dest/${pair##*:}"; cp -r "$src/." "$dest/${pair##*:}/"; fi
    done
    if [ -f "$out/methanol_pilot_ext/pull_clean/identity_filter.csv" ]; then
        cp "$out/methanol_pilot_ext/pull_clean/identity_filter.csv" "$dest/pilot_ext/"
    fi
    if [ -f "$out/identity_audit/facts_identity_audit.json" ]; then
        mkdir -p "$dest/identity_audit_v3"
        cp "$out/identity_audit/facts_identity_audit.json" "$out/identity_audit/identity_audit_years.csv" \
            "$dest/identity_audit_v3/"
    fi
    echo "results -> $dest"
}

fake() {
    local d; d="$(abspath "${1:-$ROOT/fake}")"
    local in="$d/inputs" out="$d/out"
    [ -e "$in" ] || run make_fake_inputs.py "$in"
    local gfw="$in/gfw" sw="$in/seaweb/ship_info.csv" cal="$in/type_calibration.csv"
    local rc="$in/route_cache.csv" pull="$in/methanol_pull" pd="$out/methanol_pilot"
    run build_stops.py "$out" --gfw "$gfw" --jobs "$J"
    run route_distances.py "$out" --cache "$rc" --jobs 1
    run build_stops.py "$out" --gfw "$gfw" --routes "$out/route_distances.csv" --jobs "$J"
    run nodes.py "$out"
    run vessels.py "$out" --seaweb "$sw"
    run select_ports.py "$out" --years 2019,2022 --ranges 3000,7000 --nmax 6 --candidates 20 --jobs "$J"
    run select_ports.py "$out" --years 2022 --ranges 7000 --nmax 3 --candidates 20 --jobs 1 \
        --initial-key "greedy|2019|7000|all" --initial-n 4
    run select_ports.py "$out" --years 2019 --ranges 7000 --nmax 4 --candidates 20 --jobs 1 \
        --groups container --append
    run evaluate.py "$out" --ns 2,4,6 --ranges 3000,7000,20000 --years 2019-2023 --jobs "$J" \
        --ship-years-ns 4 --weight co2 --calibration "$cal"
    run evaluate.py "$out" --ns 4 --ranges 7000 --years 2019-2023 --jobs 1 --no-ship-years \
        --slow-as-break --tag slow --only '^(greedy|volume)\|'
    run nodes.py "$out" --merge-km 15 --out nodes_km15.csv
    run select_ports.py "$out" --years 2019,2022 --ranges 7000 --nmax 6 --candidates 20 --jobs 1 \
        --nodes nodes_km15.csv --tag km15
    run evaluate.py "$out" --sets port_sets_km15.csv --nodes nodes_km15.csv --ns 4 --ranges 7000 \
        --years 2019-2023 --jobs 1 --no-ship-years --tag km15
    run select_ports.py "$out" --years 2019,2022 --ranges 7000 --nmax 6 --candidates 20 --jobs 1 \
        --tag notransit --exclude-candidates reference/transit_anchorage_ids.txt
    run evaluate.py "$out" --sets port_sets_notransit.csv --ns 4 --ranges 7000 --years 2019-2023 \
        --jobs 1 --no-ship-years --tag notransit
    run analyze.py "$out" --calibration "$cal" --train 2019 --eval 2022 --range 7000 --n 4
    run select_dualfuel.py "$out" --years 2019,2022 --range 7000 --nmax 6 --candidates 20 --jobs "$J"
    run supplement.py "$out" --calibration "$cal" --train 2019 --eval 2022 --range 7000 --n 4
    run loss_by_gap.py "$out" --n 4 --years 2019-2023 --train 2019 --eval 2022
    run persistence_dualfuel.py "$out" --calibration "$cal" --n 4 --years 2019-2023 \
        --train 2019 --eval 2022
    run identity_filter.py "$pull" "$pd/pull_clean"
    run build_stops.py "$pd" --gfw "$pd/pull_clean" --jobs 1
    run route_distances.py "$pd" --cache "$out/route_distances.csv" "$rc" --jobs 1
    run build_stops.py "$pd" --gfw "$pd/pull_clean" --routes "$pd/route_distances.csv" --jobs 1
    run pilot.py sets "$pd" --main "$out" --ships "$pull/ships.csv" --train-years 2019,2022 --nmax 6
    run evaluate.py "$pd" --ns 4 --ranges 3000,7000,20000 --years 2023 --ship-years-ns 4 --jobs 1
    run select_ports.py "$out" --years 2022 --methods "" --tag methanol --jobs 1
    run evaluate.py "$out" --sets port_sets_methanol.csv --tag methanol --only '^external:methanol6' \
        --weight co2 --calibration "$cal" --ns 4 --ranges 3000,7000,20000 --years 2019-2023 \
        --ship-years-ns 4 --jobs 1
    run pilot.py analyze "$pd" --main "$out" --ships "$pull/ships.csv" --pull "$pull" --pool 2023 \
        --main-years 2022-2023 --train 2022 --train-old 2019 --n 4
    run pilot_breakdown.py "$pd" --main "$out" --ships "$pull/ships.csv" --pool 2023 \
        --main-years 2022-2023 --train 2022 --train-old 2019 --n 4 --min-stops 3
    run methanol_dated.py "$out" --calibration "$cal" --years 2022-2023 --headline 2023 \
        --pilot "$pd" --pilot-years 2023
    run fueleu.py "$out/results"
    collect "$out" "$d/results"
}

full() {
    : "${GFW_DIR:?set GFW_DIR}" "${SEAWEB:?set SEAWEB}" "${OUT_DIR:?set OUT_DIR}"
    local out; out="$(abspath "$OUT_DIR")"
    local cal=() rc=()
    [ -n "${CALIBRATION:-}" ] && cal=(--calibration "$CALIBRATION")
    [ -n "${ROUTE_CACHE:-}" ] && rc=("$ROUTE_CACHE")
    run build_stops.py "$out" --gfw "$GFW_DIR" --jobs "$J"
    run route_distances.py "$out" --cache ${rc[@]+"${rc[@]}"} --jobs "$J"
    run build_stops.py "$out" --gfw "$GFW_DIR" --routes "$out/route_distances.csv" --jobs "$J"
    run nodes.py "$out"
    run vessels.py "$out" --seaweb "$SEAWEB"
    local c=(--years 2018-2025 --ranges 5000,7000,10000 --nmax 50 --candidates 1500 --jobs "$J")
    run select_ports.py "$out" "${c[@]}"
    run select_ports.py "$out" "${c[@]}" --groups container --append
    run select_ports.py "$out" "${c[@]}" --groups bulk --append
    run select_ports.py "$out" --years 2024 --ranges 7000 --nmax 10 --candidates 1500 \
        --initial-key "greedy|2019|7000|all" --initial-n 20 --jobs 1
    run evaluate.py "$out" --ns 5,10,20,50 --years 2018-2025 --weight co2 ${cal[@]+"${cal[@]}"} --jobs "$J" \
        --ship-years-ns 20
    local e=(--ns 20 --ranges 5000,7000,10000 --years 2018-2025 --no-ship-years --jobs "$J")
    run evaluate.py "$out" "${e[@]}" --slow-as-break --tag slow \
        --only '^(greedy|volume)\|[0-9]+\|[0-9]+\|all$'
    local km
    for km in 15 45; do
        run nodes.py "$out" --merge-km "$km" --out "nodes_km$km.csv"
        run select_ports.py "$out" --years 2019,2024 --ranges 7000 --nmax 20 --candidates 1500 \
            --nodes "nodes_km$km.csv" --tag "km$km" --jobs "$J"
        run evaluate.py "$out" "${e[@]}" --sets "port_sets_km$km.csv" --nodes "nodes_km$km.csv" --tag "km$km"
    done
    run select_ports.py "$out" --years 2019,2024 --ranges 7000 --nmax 20 --candidates 1500 \
        --exclude-candidates reference/transit_anchorage_ids.txt --tag notransit --jobs "$J"
    run evaluate.py "$out" "${e[@]}" --sets port_sets_notransit.csv --tag notransit
    run analyze.py "$out" ${cal[@]+"${cal[@]}"}
    run select_ports.py "$out" --years 2024 --methods "" --tag bunker --jobs 1
    run evaluate.py "$out" --sets port_sets_bunker.csv --tag bunker --only '^external:bunker10' \
        --weight co2 ${cal[@]+"${cal[@]}"} --ns 10 --ship-years-ns 10 --jobs "$J"
    run select_dualfuel.py "$out" --years 2019,2024 --range 7000 --nmax 20 --candidates 1500 --jobs "$J"
    run supplement.py "$out" ${cal[@]+"${cal[@]}"}
    run loss_by_gap.py "$out"
    run persistence_dualfuel.py "$out" ${cal[@]+"${cal[@]}"}
    run select_ports.py "$out" --years 2024 --methods "" --tag methanol --jobs 1
    run evaluate.py "$out" --sets port_sets_methanol.csv --tag methanol --only '^external:methanol6' \
        --weight co2 ${cal[@]+"${cal[@]}"} --ns 20 --ship-years-ns 20 --jobs "$J"
    local spec pull name ships pd
    for spec in "${METHANOL_DIR:-}|methanol_pilot|reference/methanol_ships_public.csv" \
                "${METHANOL_EXT_DIR:-}|methanol_pilot_ext|reference/methanol_ships_public_ext.csv"; do
        IFS="|" read -r pull name ships <<< "$spec"
        if ! has_pull "$pull"; then echo "--- $name skipped: no pull"; continue; fi
        pd="$out/$name"
        run identity_filter.py "$pull" "$pd/pull_clean"
        run build_stops.py "$pd" --gfw "$pd/pull_clean" --jobs "$J"
        run route_distances.py "$pd" --cache "$out/route_distances.csv" ${rc[@]+"${rc[@]}"} --jobs "$J"
        run build_stops.py "$pd" --gfw "$pd/pull_clean" --routes "$pd/route_distances.csv" --jobs "$J"
        run pilot.py sets "$pd" --main "$out" --ships "$ships"
        run evaluate.py "$pd" --ns 10,20 --years 2023-2026 --ship-years-ns 10,20 --jobs "$J"
        run pilot.py analyze "$pd" --main "$out" --pull "$pull" --ships "$ships"
        run pilot_breakdown.py "$pd" --main "$out" --ships "$ships"
    done
    run methanol_dated.py "$out" ${cal[@]+"${cal[@]}"} --pilot "$out/methanol_pilot,$out/methanol_pilot_ext"
    run fueleu.py "$out/results"
    collect "$out" "$out/bundle"
}

identity() {
    : "${OUT_DIR:?set OUT_DIR}" "${IDSAMPLE_DIR:?set IDSAMPLE_DIR}"
    local out rr; out="$(abspath "$OUT_DIR")"; rr="$out/identity_rerun"
    local cal=() rc=()
    [ -n "${CALIBRATION:-}" ] && cal=(--calibration "$CALIBRATION")
    [ -n "${ROUTE_CACHE:-}" ] && rc=("$ROUTE_CACHE")
    run identity_filter.py "$IDSAMPLE_DIR" "$rr/pull_clean"
    run build_stops.py "$rr" --gfw "$rr/pull_clean" --jobs "$J"
    run route_distances.py "$rr" --cache "$out/route_distances.csv" ${rc[@]+"${rc[@]}"} --jobs "$J"
    run build_stops.py "$rr" --gfw "$rr/pull_clean" --routes "$rr/route_distances.csv" --jobs "$J"
    cp "$out/nodes.csv" "$out/port_sets.csv" "$out/vessels.csv" "$rr/"
    run evaluate.py "$rr" --weight co2 ${cal[@]+"${cal[@]}"} --ns 20 --years 2018-2025 --ship-years-ns 20 \
        --only '^(external:yap[48]\||greedy\|(2019|2024)\|7000\|all$)' --jobs "$J"
    run identity_compare.py "$rr" --main "$out"
    collect "$out" "$out/bundle"
}

restore() {
    local list="$ROOT/LARGE_FILES.sha256" h f
    [ -f "$list" ] || { echo "nothing to restore"; return 0; }
    while read -r h f; do
        if [ -f "$ROOT/$f.part00" ]; then cat "$ROOT/$f".part* > "$ROOT/$f"; rm "$ROOT/$f".part*; fi
        if [ -f "$ROOT/$f.gz.part00" ]; then cat "$ROOT/$f".gz.part* > "$ROOT/$f.gz"; rm "$ROOT/$f".gz.part*; fi
        if [ -f "$ROOT/$f.gz" ] && [ ! -f "$ROOT/$f" ]; then gunzip "$ROOT/$f.gz"; fi
    done < "$list"
    if command -v sha256sum > /dev/null; then (cd "$ROOT" && sha256sum -c "$list")
    else (cd "$ROOT" && shasum -a 256 -c "$list"); fi
}

case "${1:-}" in
    fake) fake "${2:-}" ;;
    full) full ;;
    identity) identity ;;
    collect) : "${OUT_DIR:?set OUT_DIR}"; collect "$(abspath "$OUT_DIR")" "$(abspath "$OUT_DIR")/bundle" ;;
    restore) restore ;;
    *) echo "usage: bash run.sh fake [DIR] | full | identity | collect | restore"; exit 2 ;;
esac
