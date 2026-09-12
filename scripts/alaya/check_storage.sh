#!/usr/bin/env bash
# Run this FIRST, the moment a new Workshop opens - before cloning, before
# building anything.
#
#   bash scripts/alaya/check_storage.sh
#
# Answers one question: is there a mount here that survives the Workshop being
# released? Creating a volume on the platform and attaching it to a Workshop are
# two separate steps, and a Workshop created without the Container Path field
# filled in looks completely normal until everything disappears.

set -uo pipefail

echo "== mounts (excluding tmpfs/devtmpfs/proc-like) =="
printf '%-28s %-10s %6s %6s %s\n' "MOUNT" "TYPE" "SIZE" "AVAIL" "VERDICT"

ROOT_DEV="$(stat -c %d / 2>/dev/null)"
CANDIDATES=()

while read -r src mnt fstype opts _; do
    case "$fstype" in
        tmpfs|devtmpfs|proc|sysfs|cgroup*|devpts|mqueue|securityfs| \
        debugfs|tracefs|bpf|configfs|fusectl|pstore|autofs|binfmt_misc|nsfs)
            continue ;;
    esac
    # Skip bind-mounted single files (e.g. /etc/hostname, /etc/resolv.conf).
    [ -d "$mnt" ] || continue

    size="$(df -h --output=size "$mnt" 2>/dev/null | tail -1 | tr -d ' ')"
    avail="$(df -h --output=avail "$mnt" 2>/dev/null | tail -1 | tr -d ' ')"
    dev="$(stat -c %d "$mnt" 2>/dev/null)"

    verdict=""
    case ",$opts," in *,ro,*) verdict="read-only" ;; esac

    if [ -z "$verdict" ]; then
        if [ "$mnt" = "/" ]; then
            verdict="EPHEMERAL (container disk)"
        elif [ "$dev" = "$ROOT_DEV" ]; then
            verdict="EPHEMERAL (same device as /)"
        else
            probe="$mnt/.idm-write-test.$$"
            if touch "$probe" 2>/dev/null; then
                rm -f "$probe"
                # Network filesystems are the ones a PVC actually rides on. A
                # local block device mounted here is just as likely to be the
                # node's own scratch - on Alaya, /anc-init is 867 GB of exactly
                # that, and writing there both fails to persist and eats disk
                # your neighbours on the node are sharing.
                case "$fstype" in
                    nfs|nfs4|ceph|cephfs|glusterfs|fuse.*|9p|virtiofs)
                        verdict="PERSISTENT (network storage)"
                        CANDIDATES+=("net|$mnt|$avail|$fstype") ;;
                    *)
                        verdict="writable, but local disk - verify it is YOUR mount"
                        CANDIDATES+=("local|$mnt|$avail|$fstype") ;;
                esac
            else
                verdict="not writable"
            fi
        fi
    fi
    printf '%-28s %-10s %6s %6s %s\n' "$mnt" "$fstype" "$size" "$avail" "$verdict"
done < /proc/mounts

echo
if [ "${#CANDIDATES[@]}" -eq 0 ]; then
    cat <<'EOF'
== NO PERSISTENT STORAGE ATTACHED ==

Nothing here outlives the Workshop. Anything you build or download is lost when
the container is released.

To fix it, both steps are required - doing only the first is the usual mistake:

  1. Create the volume: platform console -> 产品中心 -> 存储管理 -> create a
     NAS/file volume, 100 GB+, in the SAME cluster as your GPU.

  2. Create a NEW Workshop in Aladdin with Storage fully filled in:
       - volume:         the one from step 1
       - capacity:       100 GB+
       - Container Path: /pvc          <- the field that is blank by default

A running Workshop cannot have storage added to it: mounts are fixed when the
pod is created. You have to make a new one.

Then re-run this script and expect /pvc to show up below.
EOF
    exit 1
fi

echo "== writable candidates =="
NET=()
LOCAL=()
for c in "${CANDIDATES[@]}"; do
    IFS='|' read -r kind mnt avail fstype <<< "$c"
    if [ "$kind" = "net" ]; then
        NET+=("$mnt"); echo "  $mnt  ($avail free, $fstype)  <- network storage, persists"
    else
        LOCAL+=("$mnt"); echo "  $mnt  ($avail free, $fstype)  <- LOCAL disk, see warning"
    fi
done

if [ "${#NET[@]}" -eq 0 ]; then
    cat <<'EOF'

WARNING: every writable candidate is a LOCAL disk, not network storage.

A PVC normally shows up as nfs/cephfs. A big local mount is usually the node's
own scratch space - it does not follow you when the pod is rescheduled, and on
a shared cluster it is shared with whoever else lands on that node. Alaya's
/anc-init is exactly this: 867 GB that looks inviting and is not yours.

Only use one of these if it is the exact path you typed into Container Path
when creating the Workshop. Otherwise treat this as "no storage attached" and
follow the two steps above.
EOF
    best="${LOCAL[0]}"
else
    best="${NET[0]}"
fi
cat <<EOF

Looks good. Use it:

    export IDM_ROOT=$best/idm
    mkdir -p "\$IDM_ROOT"

Caveat worth knowing: "writable and on its own device" is strong evidence, not
proof. The only real proof is that a file written now is still there after the
Workshop is released and recreated. Before committing to a long download, write
a marker and check for it next session:

    echo "written \$(date -Is)" > $best/.idm-persistence-check
EOF
